import argparse
import csv
import json
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

from sync.v1 import FixedGuardSyncV1, SyncV1Config, makeGuardBank


RATE = 16000
FRAME = 320
CORE = 240
GUARD = 80

TX_SINK = "tx_enc_sink"
RX_SINK = "rx_sink"
TX_MIC = "tx_mic"


def pactl(*args):
    return subprocess.check_output(
        ["pactl", *args],
        text=True
    ).strip()


def getDefaultSource():
    info = pactl("info")

    for line in info.splitlines():
        if line.startswith("Default Source:"):
            return line.split(":", 1)[1].strip()

    raise RuntimeError("Default Source not found in pactl info")


def getDefaultSink():
    info = pactl("info")

    for line in info.splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip()

    raise RuntimeError("Default Sink not found in pactl info")


class RealtimeVirtualAudio:
    def __init__(self):
        self.modules = []

    def _cleanupStale(self):
        text = pactl("list", "short", "modules")

        for line in reversed(text.splitlines()):
            if (
                "sink_name=tx_enc_sink" in line
                or "sink_name=rx_sink" in line
                or "source_name=tx_mic" in line
            ):
                moduleId = line.split()[0]

                subprocess.run(
                    ["pactl", "unload-module", moduleId],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

    def _load(self, *args):
        moduleId = pactl(
            "load-module",
            *args
        )

        self.modules.append(moduleId)
        return moduleId

    def start(self):
        self._cleanupStale()

        self._load(
            "module-null-sink",
            f"sink_name={TX_SINK}",
            "sink_properties=device.description=NearS_TX"
        )

        self._load(
            "module-null-sink",
            f"sink_name={RX_SINK}",
            "sink_properties=device.description=NearS_RX"
        )

        self._load(
            "module-remap-source",
            f"master={TX_SINK}.monitor",
            f"source_name={TX_MIC}",
            "source_properties=device.description=NearS_TX_Microphone"
        )

    def stop(self):
        for moduleId in reversed(self.modules):
            subprocess.run(
                ["pactl", "unload-module", moduleId],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        self.modules.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

def pcmToFloat(data):
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def floatToPcm(x):
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def readExact(stream, n):
    chunks = []
    got = 0

    while got < n:
        data = stream.read(n - got)

        if not data:
            return b"".join(chunks)

        chunks.append(data)
        got += len(data)

    return b"".join(chunks)


class WavWriter:
    def __init__(self, path):
        self.w = wave.open(str(path), "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(RATE)

    def write(self, x):
        self.w.writeframes(floatToPcm(x))

    def close(self):
        self.w.close()


class RealtimePlayout:
    def __init__(self, sink, delayMs, log):
        self.sink = sink
        self.delaySec = delayMs / 1000.0
        self.log = log

        self.frames = {}
        self.lock = threading.Lock()
        self.stopEvent = threading.Event()

        self.firstFrame = None
        self.nextFrame = None
        self.startedAt = None

        self.proc = subprocess.Popen(
            [
                "pacat",
                f"--device={sink}",
                "--format=s16le",
                f"--rate={RATE}",
                "--channels=1",
                "--latency-msec=20"
            ],
            stdin=subprocess.PIPE,
            bufsize=0
        )

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self.thread.start()

    def submit(self, frameIndex, frame):
        with self.lock:
            if self.firstFrame is None:
                self.firstFrame = frameIndex
                self.nextFrame = frameIndex
                self.startedAt = time.monotonic() + self.delaySec

            if frameIndex >= self.nextFrame:
                self.frames[frameIndex] = frame

    def _loop(self):
        zero = np.zeros(FRAME, dtype=np.float32)

        while not self.stopEvent.is_set():
            with self.lock:
                startedAt = self.startedAt
                nextFrame = self.nextFrame

            if startedAt is None:
                time.sleep(0.005)
                continue

            due = startedAt + (
                nextFrame - self.firstFrame
            ) * (FRAME / RATE)

            wait = due - time.monotonic()

            if wait > 0:
                time.sleep(min(wait, 0.005))
                continue

            with self.lock:
                frame = self.frames.pop(
                    nextFrame,
                    None
                )

                self.nextFrame += 1

            if frame is None:
                frame = zero

                self.log(
                    "playout_missing",
                    frame=nextFrame
                )

            self.proc.stdin.write(
                floatToPcm(frame)
            )

    def close(self):
        self.stopEvent.set()
        self.thread.join(timeout=2)

        try:
            self.proc.stdin.close()
        except Exception:
            pass

        self.proc.terminate()

        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def runTx(args):
    guards = makeGuardBank(
        RATE,
        GUARD,
        args.spinner_size
    )

    source = subprocess.Popen(
        [
            "parec",
            f"--device={args.source}",
            "--format=s16le",
            f"--rate={RATE}",
            "--channels=1",
            "--latency-msec=20"
        ],
        stdout=subprocess.PIPE,
        bufsize=0
    )

    sink = subprocess.Popen(
        [
            "pacat",
            f"--device={args.sink}",
            "--format=s16le",
            f"--rate={RATE}",
            "--channels=1",
            "--latency-msec=20"
        ],
        stdin=subprocess.PIPE,
        bufsize=0
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rawWav = WavWriter(out / "tx_raw.wav")
    syncWav = WavWriter(out / "tx_sync.wav")

    logFile = open(out / "events.jsonl", "w")

    def log(event, **fields):
        row = {
            "monotonic": time.monotonic(),
            "event": event,
            **fields
        }

        logFile.write(json.dumps(row) + "\n")
        logFile.flush()

    frameIndex = 0
    nextStatus = time.monotonic()

    print("REALTIME SYNC TX RUNNING")
    print(f"source={args.source}")
    print(f"sink={args.sink}")

    try:
        while True:
            data = readExact(
                source.stdout,
                FRAME * 2
            )

            if len(data) != FRAME * 2:
                break

            x = pcmToFloat(data)

            phase = frameIndex % args.spinner_size

            rms = float(
                np.sqrt(np.mean(x * x) + 1e-12)
            )

            floor = 10.0 ** (
                args.guard_floor_dbfs / 20.0
            )

            guardAmp = max(
                rms * args.guard_ratio,
                floor
            )

            y = x.copy()

            y[CORE:] += (
                guards[phase] * guardAmp
            )

            clipped = int(
                np.sum(np.abs(y) > 1.0)
            )

            y = np.clip(
                y,
                -0.999,
                0.999
            )

            rawWav.write(x)
            syncWav.write(y)

            sink.stdin.write(
                floatToPcm(y)
            )

            if clipped:
                log(
                    "tx_clip",
                    frame=frameIndex,
                    phase=phase,
                    clipped=clipped
                )

            now = time.monotonic()

            if now >= nextStatus:
                print(
                    f"\rTX frame={frameIndex:7d} "
                    f"spin={phase} "
                    f"rms={rms:.4f} "
                    f"pnAmp={guardAmp:.4f} "
                    f"clip={clipped}",
                    end="",
                    flush=True
                )

                nextStatus = now + 1.0

            frameIndex += 1

    except KeyboardInterrupt:
        print("\nTX stopped")

    finally:
        rawWav.close()
        syncWav.close()
        logFile.close()

        source.terminate()
        sink.terminate()

        print(f"\nTX frames={frameIndex}")


def runRx(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    receivedWav = WavWriter(
        out / "rx_raw.wav"
    )

    recoveredWav = WavWriter(
        out / "rx_recovered.wav"
    )

    framesFile = open(
        out / "frames.csv",
        "w",
        newline=""
    )

    framesCsv = csv.writer(framesFile)

    framesCsv.writerow([
        "frame",
        "status",
        "state",
        "phase",
        "guard_pos",
        "corr",
        "timing_error_samples",
        "timing_error_ms",
        "observed_period",
        "period_estimate",
        "drift_ppm",
        "predicted_frame",
        "core_input_samples",
        "core_correction_samples",
        "pn_removed_gain"
    ])

    eventsFile = open(
        out / "events.jsonl",
        "w"
    )

    source = subprocess.Popen(
        [
            "parec",
            f"--device={args.source}",
            "--format=s16le",
            f"--rate={RATE}",
            "--channels=1",
            "--latency-msec=10"
        ],
        stdout=subprocess.PIPE,
        bufsize=0
    )

    sync = None
    playout = None
    previousGuard = None
    nextStatus = 0.0
    latest = {}

    def log(event, **fields):
        row = {
            "monotonic": time.monotonic(),
            "event": event,
            **fields
        }

        eventsFile.write(
            json.dumps(row) + "\n"
        )

        eventsFile.flush()

        if event != "playout_missing":
            print(
                f"\n[{event.upper()}] "
                + " ".join(
                    f"{k}={v}"
                    for k, v in fields.items()
                )
            )

    def normalizeCore(x):
        if len(x) == CORE:
            return x.copy()

        if len(x) < 2:
            return np.zeros(
                CORE,
                dtype=np.float32
            )

        old = np.linspace(
            0.0,
            1.0,
            len(x)
        )

        new = np.linspace(
            0.0,
            1.0,
            CORE
        )

        return np.interp(
            new,
            old,
            x
        ).astype(np.float32)

    def recoverFrame(frameIndex, status, meta):
        nonlocal previousGuard

        guardPos = meta.get(
            "guard_pos"
        )

        if status != "ok" or guardPos == "":
            return np.zeros(
                FRAME,
                dtype=np.float32
            ), "", "", ""

        guardPos = int(guardPos)

        if previousGuard is None:
            coreRaw = sync.buf.slice(
                guardPos - CORE,
                guardPos
            )
        else:
            physicalPeriod = (
                guardPos - previousGuard
            )

            # Normal locked path:
            # previous guard ended exactly at the
            # beginning of this frame.
            if (
                abs(
                    physicalPeriod - FRAME
                )
                <= args.max_realtime_adjust
            ):
                coreRaw = sync.buf.slice(
                    previousGuard + GUARD,
                    guardPos
                )
            else:
                # После reacquire не тянем через gap.
                coreRaw = sync.buf.slice(
                    guardPos - CORE,
                    guardPos
                )

        guardRaw = sync.buf.slice(
            guardPos,
            guardPos + GUARD
        )

        if (
            coreRaw is None
            or guardRaw is None
        ):
            return np.zeros(
                FRAME,
                dtype=np.float32
            ), "", "", ""

        coreInput = len(coreRaw)
        coreCorrection = CORE - coreInput

        core = normalizeCore(coreRaw)

        phase = (
            frameIndex
            % args.spinner_size
        )

        pn = sync.guards[phase]

        alpha = float(
            np.dot(guardRaw, pn)
            / (
                np.dot(pn, pn)
                + 1e-12
            )
        )

        guardClean = (
            guardRaw - alpha * pn
        ).astype(np.float32)

        y = np.concatenate(
            (core, guardClean)
        )

        previousGuard = guardPos

        return (
            y,
            coreInput,
            coreCorrection,
            alpha
        )

    def onFrame(
        frameIndex,
        _unusedFrame,
        status,
        meta
    ):
        nonlocal previousGuard
        nonlocal nextStatus
        nonlocal latest

        if status == "lost":
            y = np.zeros(
                FRAME,
                dtype=np.float32
            )

            coreInput = ""
            correction = ""
            alpha = ""

        else:
            (
                y,
                coreInput,
                correction,
                alpha
            ) = recoverFrame(
                frameIndex,
                status,
                meta
            )

        if meta.get("guard_pos") not in (
            None,
            ""
        ):
            previousGuard = int(
                meta["guard_pos"]
            )

        playout.submit(
            frameIndex,
            y
        )

        recoveredWav.write(y)

        err = meta.get(
            "timing_error_samples",
            ""
        )

        errMs = (
            float(err) * 1000.0 / RATE
            if err not in ("", None)
            else ""
        )

        period = meta.get(
            "period_estimate",
            sync.periodEstimate
        )

        try:
            driftPpm = (
                float(period) / FRAME
                - 1.0
            ) * 1e6
        except Exception:
            driftPpm = ""

        phase = (
            frameIndex
            % args.spinner_size
        )

        framesCsv.writerow([
            frameIndex,
            status,
            sync.state,
            phase,
            meta.get("guard_pos", ""),
            meta.get("corr", ""),
            err,
            errMs,
            meta.get("observed_period", ""),
            period,
            driftPpm,
            meta.get("predicted_frame", ""),
            coreInput,
            correction,
            alpha
        ])

        if (
            correction not in ("", 0)
            and abs(int(correction))
            >= args.adjust_log_samples
        ):
            log(
                "adjust",
                frame=frameIndex,
                phase=phase,
                core_input_samples=coreInput,
                correction_samples=correction,
                correction_ms=(
                    correction * 1000.0 / RATE
                ),
                timing_error_samples=err,
                timing_error_ms=errMs
            )

        latest = {
            "frame": frameIndex,
            "status": status,
            "phase": phase,
            "corr": meta.get("corr", ""),
            "err": err,
            "errMs": errMs,
            "period": period,
            "driftPpm": driftPpm,
            "pred": meta.get(
                "predicted_frame",
                ""
            ),
            "correction": correction
        }

        now = time.monotonic()

        if now >= nextStatus:
            print(
                "\rRX "
                f"state={sync.state:<9} "
                f"frame={frameIndex:7d} "
                f"spin={phase} "
                f"corr={str(latest['corr'])[:6]:>6} "
                f"err={str(errMs)[:7]:>7}ms "
                f"period={float(period):7.3f} "
                f"drift={float(driftPpm):+8.1f}ppm "
                f"pred={str(latest['pred'])[:9]:>9} "
                f"fix={str(correction):>4}",
                end="",
                flush=True
            )

            nextStatus = now + 1.0

    def onEvent(event):
        fields = dict(event)

        eventName = fields.pop(
            "event",
            "sync"
        )

        fields.pop(
            "state",
            None
        )

        log(
            eventName,
            state=sync.state,
            **fields
        )

    cfg = SyncV1Config(
        rate=RATE,
        spinnerSize=args.spinner_size,
        acquireThreshold=args.acquire_threshold,
        trackThreshold=args.track_threshold
    )

    sync = FixedGuardSyncV1(
        cfg,
        onFrame,
        onEvent
    )

    playout = RealtimePlayout(
        args.sink,
        args.playout_delay_ms,
        log
    )

    print("REALTIME SYNC RX RUNNING")
    print(f"source={args.source}")
    print(f"sink={args.sink}")
    print(
        f"playout delay="
        f"{args.playout_delay_ms} ms"
    )

    try:
        while True:
            data = readExact(
                source.stdout,
                args.io_samples * 2
            )

            if not data:
                break

            x = pcmToFloat(data)

            receivedWav.write(x)

            sync.push(
                x,
                time.monotonic()
            )

    except KeyboardInterrupt:
        print("\nRX stopped")

    finally:
        playout.close()

        receivedWav.close()
        recoveredWav.close()

        framesFile.close()
        eventsFile.close()

        source.terminate()

        print()
        print(
            json.dumps(
                sync.stats,
                indent=2
            )
        )

def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--role",
        choices=["tx", "rx"],
        required=True
    )

    p.add_argument(
        "--out",
        required=True
    )

    p.add_argument(
        "--spinner-size",
        type=int,
        default=8
    )

    p.add_argument(
        "--guard-ratio",
        type=float,
        default=0.35
    )

    p.add_argument(
        "--guard-floor-dbfs",
        type=float,
        default=-30.0
    )

    p.add_argument(
        "--acquire-threshold",
        type=float,
        default=0.20
    )

    p.add_argument(
        "--track-threshold",
        type=float,
        default=0.14
    )

    p.add_argument(
        "--io-samples",
        type=int,
        default=160
    )

    p.add_argument(
        "--playout-delay-ms",
        type=float,
        default=200.0
    )

    p.add_argument(
        "--max-realtime-adjust",
        type=int,
        default=40
    )

    p.add_argument(
        "--adjust-log-samples",
        type=int,
        default=2
    )

    args = p.parse_args()

    # ВАЖНО:
    # запоминаем реальные устройства ДО создания virtual devices.
    physicalSource = getDefaultSource()
    physicalSink = getDefaultSink()

    print()
    print("Physical audio detected:")
    print(f"  microphone : {physicalSource}")
    print(f"  headphones : {physicalSink}")

    with RealtimeVirtualAudio():
        print()
        print("Virtual devices ready.")

        if args.role == "tx":
            print()
            print("TELEMOST TX SETTINGS:")
            print(f"  Microphone = {TX_MIC}")
            print()
            print("Physical microphone used by synchronizer:")
            print(f"  {physicalSource}")
            print()

            input("Join Telemost, select microphone, then press ENTER to start TX... ")

            args.source = physicalSource
            args.sink = TX_SINK

            runTx(args)

        else:
            print()
            print("TELEMOST RX SETTINGS:")
            print(f"  Speakers = {RX_SINK}")
            print()
            print("Physical headphones used for corrected audio:")
            print(f"  {physicalSink}")
            print()

            input("Join Telemost, select speakers, then press ENTER to start RX... ")

            args.source = f"{RX_SINK}.monitor"
            args.sink = physicalSink

            runRx(args)

if __name__ == "__main__":
    main()