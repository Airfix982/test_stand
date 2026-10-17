import argparse, csv, json, subprocess, threading, time, wave
from pathlib import Path
import numpy as np
from sync.v2 import FixedGuardSyncV1, SyncV1Config, makeGuardBank

RATE, FRAME, CORE, GUARD = 16000, 320, 240, 80
TX_SINK, RX_SINK, TX_MIC = "tx_enc_sink", "rx_sink", "tx_mic"

def pactl(*args):
    return subprocess.check_output(["pactl", *args], text=True).strip()

def getDefault(kind):
    prefix = "Default Source:" if kind == "source" else "Default Sink:"
    for line in pactl("info").splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"{prefix[:-1]} not found in pactl info")

def pcmToFloat(data):
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0

def floatToPcm(x):
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

def readExact(stream, n):
    chunks, got = [], 0
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
        self.w.setnchannels(1); self.w.setsampwidth(2); self.w.setframerate(RATE)
    def write(self, x): self.w.writeframes(floatToPcm(x))
    def close(self): self.w.close()

class RealtimeVirtualAudio:
    def __init__(self, role):
        self.role, self.modules = role, []

    def _matches(self, line):
        if self.role == "tx":
            return "sink_name=tx_enc_sink" in line or "source_name=tx_mic" in line
        return "sink_name=rx_sink" in line

    def _cleanup(self):
        for line in reversed(pactl("list", "short", "modules").splitlines()):
            if self._matches(line):
                subprocess.run(["pactl", "unload-module", line.split()[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _load(self, *args):
        module = pactl("load-module", *args)
        self.modules.append(module)

    def start(self):
        self._cleanup()
        if self.role == "tx":
            self._load("module-null-sink", f"sink_name={TX_SINK}", "sink_properties=device.description=NearS_TX")
            self._load("module-remap-source", f"master={TX_SINK}.monitor", f"source_name={TX_MIC}", "source_properties=device.description=NearS_TX_Microphone")
        else:
            self._load("module-null-sink", f"sink_name={RX_SINK}", "sink_properties=device.description=NearS_RX")

    def stop(self):
        for module in reversed(self.modules):
            subprocess.run(["pactl", "unload-module", module], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.modules.clear()

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()

class RealtimePlayout:
    def __init__(self, sink, delayMs, log):
        self.delaySec, self.log = delayMs / 1000.0, log
        self.frames, self.lock, self.stopEvent = {}, threading.Lock(), threading.Event()
        self.firstFrame = self.nextFrame = self.startedAt = None
        self.missingStart, self.missingCount = None, 0
        self.proc = subprocess.Popen(["pacat", f"--device={sink}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdin=subprocess.PIPE, bufsize=0)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frameIndex, frame):
        with self.lock:
            if self.firstFrame is None:
                self.firstFrame = self.nextFrame = frameIndex
                self.startedAt = time.monotonic() + self.delaySec
            if frameIndex >= self.nextFrame:
                self.frames[frameIndex] = frame.copy()

    def _loop(self):
        zero = np.zeros(FRAME, dtype=np.float32)
        while not self.stopEvent.is_set():
            with self.lock:
                startedAt, nextFrame = self.startedAt, self.nextFrame
            if startedAt is None:
                time.sleep(0.005); continue
            wait = startedAt + (nextFrame - self.firstFrame) * FRAME / RATE - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 0.005)); continue
            with self.lock:
                frame = self.frames.pop(nextFrame, None)
                self.nextFrame += 1
            if frame is None:
                frame = zero
                if self.missingStart is None: self.missingStart = nextFrame
                self.missingCount += 1
            elif self.missingStart is not None:
                self.log("playout_gap", start_frame=self.missingStart, end_frame=nextFrame - 1, frames=self.missingCount, duration_ms=self.missingCount * 20)
                self.missingStart, self.missingCount = None, 0
            try:
                self.proc.stdin.write(floatToPcm(frame))
            except (BrokenPipeError, OSError):
                break

    def close(self):
        self.stopEvent.set(); self.thread.join(timeout=2)
        try: self.proc.stdin.close()
        except Exception: pass
        self.proc.terminate()
        try: self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()

def runTx(args):
    guards = makeGuardBank(RATE, GUARD, args.spinner_size)
    source = subprocess.Popen(["parec", f"--device={args.source}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdout=subprocess.PIPE, bufsize=0)
    sink = subprocess.Popen(["pacat", f"--device={args.sink}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdin=subprocess.PIPE, bufsize=0)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rawWav, syncWav = WavWriter(out / "tx_raw.wav"), WavWriter(out / "tx_sync.wav")
    logFile = open(out / "events.jsonl", "w")

    def log(event, **fields):
        logFile.write(json.dumps({"monotonic": time.monotonic(), "event": event, **fields}) + "\n"); logFile.flush()

    frameIndex, nextStatus = 0, time.monotonic()
    print(f"REALTIME V2 TX RUNNING\nsource={args.source}\nsink={args.sink}")
    try:
        while True:
            data = readExact(source.stdout, FRAME * 2)
            if len(data) != FRAME * 2: break
            x = pcmToFloat(data)
            phase = frameIndex % args.spinner_size
            rms = float(np.sqrt(np.mean(x[:CORE] * x[:CORE]) + 1e-12))
            floor = 10.0 ** (args.guard_floor_dbfs / 20.0)
            guardAmp = max(rms * args.guard_ratio, floor)

            # V2 wire format: 240 payload samples + 80 pure guard samples.
            y = np.empty(FRAME, dtype=np.float32)
            y[:CORE] = x[:CORE]
            y[CORE:] = guards[phase] * guardAmp
            clipped = int(np.sum(np.abs(y) > 1.0))
            y = np.clip(y, -0.999, 0.999)

            rawWav.write(x); syncWav.write(y); sink.stdin.write(floatToPcm(y))
            if clipped: log("tx_clip", frame=frameIndex, phase=phase, clipped=clipped)
            now = time.monotonic()
            if now >= nextStatus:
                log("tx_status", frame=frameIndex, phase=phase, rms=rms, pn_amp=guardAmp, clipped=clipped)
                print(f"\rTX frame={frameIndex:7d} spin={phase} rms={rms:.4f} pnAmp={guardAmp:.4f} clip={clipped}", end="", flush=True)
                nextStatus = now + 1.0
            frameIndex += 1
    except KeyboardInterrupt:
        print("\nTX stopped")
    finally:
        rawWav.close(); syncWav.close(); logFile.close()
        source.terminate(); sink.terminate()
        print(f"\nTX frames={frameIndex}")

def runRx(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    receivedWav, recoveredWav = WavWriter(out / "rx_raw.wav"), WavWriter(out / "rx_recovered.wav")
    framesFile = open(out / "frames.csv", "w", newline="")
    framesCsv = csv.writer(framesFile)
    framesCsv.writerow(["frame", "status", "state", "phase", "guard_pos", "corr", "timing_error_samples", "timing_error_ms", "observed_period", "period_estimate", "drift_ppm", "predicted_frame", "reason"])
    eventsFile = open(out / "events.jsonl", "w")
    source = subprocess.Popen(["parec", f"--device={args.source}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=10"], stdout=subprocess.PIPE, bufsize=0)
    sync = playout = None
    nextStatus = 0.0

    def log(event, **fields):
        eventsFile.write(json.dumps({"monotonic": time.monotonic(), "event": event, **fields}) + "\n"); eventsFile.flush()
        if event != "playout_missing":
            print("\n[" + event.upper() + "] " + " ".join(f"{k}={v}" for k, v in fields.items()))

    def onFrame(frameIndex, frame, status, meta):
        nonlocal nextStatus
        # sync.v2 already returns a normalized 240-sample recovered core in frame[:240].
        # For this sync-only Telemost test the final 80 samples stay zero.
        y = frame if status == "ok" else np.zeros(FRAME, dtype=np.float32)
        recoveredWav.write(y); playout.submit(frameIndex, y)

        err = meta.get("timing_error_samples", "")
        errMs = float(err) * 1000.0 / RATE if err not in ("", None) else ""
        period = meta.get("period_estimate", sync.periodEstimate)
        try: driftPpm = (float(period) / FRAME - 1.0) * 1e6
        except Exception: driftPpm = ""
        phase = frameIndex % args.spinner_size
        framesCsv.writerow([frameIndex, status, sync.state, phase, meta.get("guard_pos", ""), meta.get("corr", ""), err, errMs, meta.get("observed_period", ""), period, driftPpm, meta.get("predicted_frame", ""), meta.get("reason", "")])
        framesFile.flush()

        now = time.monotonic()
        if now >= nextStatus:
            corr = str(meta.get("corr", ""))[:6]
            pred = str(meta.get("predicted_frame", ""))[:9]
            print(f"\rRX state={sync.state:<9} frame={frameIndex:7d} spin={phase} corr={corr:>6} err={str(errMs)[:7]:>7}ms period={float(period):7.3f} drift={float(driftPpm):+8.1f}ppm pred={pred:>9}", end="", flush=True)
            nextStatus = now + 1.0

    def onEvent(event):
        fields = dict(event)
        eventName = fields.pop("event", "sync")
        fields.pop("state", None)
        log(eventName, state=sync.state, **fields)

    cfg = SyncV1Config(rate=RATE, spinnerSize=args.spinner_size, acquireThreshold=args.acquire_threshold, trackThreshold=args.track_threshold)
    sync = FixedGuardSyncV1(cfg, onFrame, onEvent)
    playout = RealtimePlayout(args.sink, args.playout_delay_ms, log)
    print(f"REALTIME V2 RX RUNNING\nsource={args.source}\nsink={args.sink}\nplayout delay={args.playout_delay_ms} ms")

    try:
        while True:
            data = readExact(source.stdout, args.io_samples * 2)
            if not data: break
            x = pcmToFloat(data)
            receivedWav.write(x)
            sync.push(x, time.monotonic())
    except KeyboardInterrupt:
        print("\nRX stopped")
    finally:
        try: sync.finish()
        except Exception as e: log("finish_error", error=repr(e))
        playout.close(); receivedWav.close(); recoveredWav.close()
        framesFile.close(); eventsFile.close(); source.terminate()
        print("\n" + json.dumps(sync.stats, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["tx", "rx"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--spinner-size", type=int, default=8)
    p.add_argument("--guard-ratio", type=float, default=0.8)
    p.add_argument("--guard-floor-dbfs", type=float, default=-24.0)
    p.add_argument("--acquire-threshold", type=float, default=0.28)
    p.add_argument("--track-threshold", type=float, default=0.20)
    p.add_argument("--io-samples", type=int, default=160)
    p.add_argument("--playout-delay-ms", type=float, default=250.0)
    args = p.parse_args()

    physicalSource, physicalSink = getDefault("source"), getDefault("sink")
    print(f"Physical audio:\n  microphone : {physicalSource}\n  headphones : {physicalSink}")

    with RealtimeVirtualAudio(args.role):
        if args.role == "tx":
            print(f"\nTELEMOST TX: select microphone '{TX_MIC}' / NearS_TX_Microphone")
            input("Join Telemost and select it, then press ENTER to start TX... ")
            args.source, args.sink = physicalSource, TX_SINK
            runTx(args)
        else:
            print(f"\nTELEMOST RX: select speakers '{RX_SINK}' / NearS_RX")
            input("Join Telemost and select it, then press ENTER to start RX... ")
            args.source, args.sink = f"{RX_SINK}.monitor", physicalSink
            runRx(args)

if __name__ == "__main__":
    main()