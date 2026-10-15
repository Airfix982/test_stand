import argparse
import csv
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from sync.v2 import FixedGuardSyncV1, SyncV1Config, makeGuardBank


RATE = 16000
FRAME = 320
CORE = 240
GUARD = 80

TX_SINK = "syncprobe_tx_sink"
TX_MIC = "syncprobe_tx_mic"
RX_SINK = "syncprobe_rx_sink"

BROWSER_NAMES = ("chrome", "chromium", "firefox", "yandex")


def pactl(*args):
    return subprocess.check_output(["pactl", *args], text=True).strip()


def get_default(key):
    for line in pactl("info").splitlines():
        if line.startswith(key + ":"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"{key} not found in pactl info")


def cleanup_named_modules(names):
    text = pactl("list", "short", "modules")
    for line in reversed(text.splitlines()):
        if any(name in line for name in names):
            subprocess.run(
                ["pactl", "unload-module", line.split()[0]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


class Modules:
    def __init__(self):
        self.ids = []

    def load(self, *args):
        mid = pactl("load-module", *args)
        self.ids.append(mid)
        return mid

    def close(self):
        for mid in reversed(self.ids):
            subprocess.run(
                ["pactl", "unload-module", mid],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.ids.clear()


def index_for_name(kind, name):
    for line in pactl("list", "short", kind).splitlines():
        cols = line.split()
        if len(cols) >= 2 and cols[1] == name:
            return int(cols[0])
    raise RuntimeError(f"{kind}: {name} not found")


def browser_streams(kind):
    text = pactl("list", kind)

    if kind == "source-outputs":
        header = "Source Output #"
        route_key = "Source:"
    elif kind == "sink-inputs":
        header = "Sink Input #"
        route_key = "Sink:"
    else:
        raise ValueError(kind)

    rows = []
    cur = None

    def finish():
        nonlocal cur
        if cur is not None:
            app = (cur.get("application") or "").lower()
            if any(name in app for name in BROWSER_NAMES):
                rows.append(cur)
        cur = None

    for raw in text.splitlines():
        s = raw.strip()

        if s.startswith(header):
            finish()
            cur = {"id": int(s.split("#", 1)[1])}
            continue

        if cur is None:
            continue

        if s.startswith(route_key):
            cur["route_index"] = int(s.split(":", 1)[1].strip())
        elif s.startswith('application.name = "'):
            cur["application"] = s.split("=", 1)[1].strip().strip('"')
        elif s.startswith('media.name = "'):
            cur["media"] = s.split("=", 1)[1].strip().strip('"')

    finish()
    return rows


def route_tx_browser():
    rows = browser_streams("source-outputs")
    if not rows:
        raise RuntimeError("No live browser microphone stream found. Join the Telemost call first.")

    for row in rows:
        print(f"Moving browser capture #{row['id']} ({row.get('application')}, {row.get('media')}) -> {TX_MIC}")
        subprocess.check_call(["pactl", "move-source-output", str(row["id"]), TX_MIC])

    time.sleep(0.25)

    expected = index_for_name("sources", TX_MIC)
    rows = browser_streams("source-outputs")
    bad = False

    for row in rows:
        ok = row.get("route_index") == expected
        print(f"TX ROUTE {'OK' if ok else 'BAD'}: browser #{row['id']} source={row.get('route_index')} expected={expected}")
        bad |= not ok

    if bad:
        raise RuntimeError("TX browser capture is not routed to the virtual microphone")


def route_rx_browser():
    rows = browser_streams("sink-inputs")
    if not rows:
        raise RuntimeError("No live browser playback stream found. Join the Telemost call first.")

    for row in rows:
        print(f"Moving browser playback #{row['id']} ({row.get('application')}, {row.get('media')}) -> {RX_SINK}")
        subprocess.check_call(["pactl", "move-sink-input", str(row["id"]), RX_SINK])

    time.sleep(0.25)

    expected = index_for_name("sinks", RX_SINK)
    rows = browser_streams("sink-inputs")
    bad = False

    for row in rows:
        ok = row.get("route_index") == expected
        print(f"RX ROUTE {'OK' if ok else 'BAD'}: browser #{row['id']} sink={row.get('route_index')} expected={expected}")
        bad |= not ok

    if bad:
        raise RuntimeError("RX browser playback is not routed to the virtual sink")


def read_exact(stream, n):
    out = bytearray()
    while len(out) < n:
        part = stream.read(n - len(out))
        if not part:
            break
        out.extend(part)
    return bytes(out)


def pcm_to_float(data):
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(x):
    return np.round(np.clip(x, -0.999, 0.999) * 32767.0).astype("<i2").tobytes()


class WavWriter:
    def __init__(self, path):
        self.w = wave.open(str(path), "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(RATE)

    def write(self, x):
        self.w.writeframes(float_to_pcm(x))

    def close(self):
        self.w.close()


def run_tx(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    physical_source = get_default("Default Source")
    print(f"TX physical microphone: {physical_source}")

    cleanup_named_modules((f"sink_name={TX_SINK}", f"source_name={TX_MIC}"))
    modules = Modules()

    raw_wav = None
    sync_wav = None
    events = None
    src = None
    dst = None

    try:
        modules.load("module-null-sink", f"sink_name={TX_SINK}", "sink_properties=device.description=SyncProbe_TX")
        modules.load("module-remap-source", f"master={TX_SINK}.monitor", f"source_name={TX_MIC}", "source_properties=device.description=SyncProbe_TX_Microphone")

        print(f"TX virtual microphone ready: {TX_MIC}")
        input("TX COMPUTER: join the Telemost call in Chrome; start RX probe first; then press ENTER here: ")

        route_tx_browser()

        src = subprocess.Popen(
            ["parec", f"--device={physical_source}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"],
            stdout=subprocess.PIPE,
            bufsize=0,
        )

        dst = subprocess.Popen(
            ["pacat", f"--device={TX_SINK}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"],
            stdin=subprocess.PIPE,
            bufsize=0,
        )

        raw_wav = WavWriter(out / "tx_mic_raw.wav")
        sync_wav = WavWriter(out / "tx_sync_stream.wav")
        events = open(out / "events.jsonl", "w")

        guards = makeGuardBank(RATE, GUARD, args.spinner_size)

        frame_index = 0
        next_print = time.monotonic()

        print("TX SYNC CHANNEL PROBE RUNNING")
        print(f"path: physical mic -> Python -> {TX_SINK} -> {TX_MIC} -> Chrome/Telemost")
        print(f"guard_amp={args.guard_amp:.4f}; core=first {CORE} mic samples; guard={GUARD} samples REPLACED by PN")
        print("Press Ctrl+C to stop TX.")

        while True:
            data = read_exact(src.stdout, FRAME * 2)
            if len(data) != FRAME * 2:
                break

            x = pcm_to_float(data)
            phase = frame_index % args.spinner_size

            y = np.empty(FRAME, dtype=np.float32)
            y[:CORE] = x[:CORE]
            y[CORE:] = guards[phase] * args.guard_amp
            y = np.clip(y, -0.999, 0.999)

            raw_wav.write(x)
            sync_wav.write(y)
            dst.stdin.write(float_to_pcm(y))

            now = time.monotonic()
            if now >= next_print:
                rms = float(np.sqrt(np.mean(x * x) + 1e-12))
                row = {
                    "monotonic": now,
                    "event": "tx_status",
                    "frame": frame_index,
                    "phase": phase,
                    "mic_rms": rms,
                    "guard_amp": args.guard_amp,
                }
                events.write(json.dumps(row) + "\n")
                events.flush()
                print(f"TX frame={frame_index:7d} phase={phase} mic_rms={rms:.5f} guard_amp={args.guard_amp:.4f}")
                next_print = now + 1.0

            frame_index += 1

    except KeyboardInterrupt:
        print("TX stopped")
    finally:
        if raw_wav:
            raw_wav.close()
        if sync_wav:
            sync_wav.close()
        if events:
            events.close()
        if src:
            src.terminate()
        if dst:
            dst.terminate()
        modules.close()


def run_rx(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cleanup_named_modules((f"sink_name={RX_SINK}",))
    modules = Modules()

    raw_wav = None
    events = None
    frames_file = None
    src = None

    latest = {
        "frame": None,
        "status": None,
        "corr": None,
        "err": None,
        "pred": None,
        "phase": None,
    }

    try:
        modules.load("module-null-sink", f"sink_name={RX_SINK}", "sink_properties=device.description=SyncProbe_RX")

        print(f"RX virtual speaker ready: {RX_SINK}")
        input("RX COMPUTER: join the Telemost call in Chrome, then press ENTER here BEFORE starting TX probe: ")

        route_rx_browser()

        raw_wav = WavWriter(out / "rx_raw.wav")
        events = open(out / "events.jsonl", "w")
        frames_file = open(out / "frames.csv", "w", newline="")
        frames_csv = csv.writer(frames_file)
        frames_csv.writerow([
            "frame", "status", "state", "phase", "guard_pos", "corr",
            "timing_error_samples", "observed_period", "period_estimate",
            "predicted_frame"
        ])

        cfg = SyncV1Config(
            spinnerSize=args.spinner_size,
            acquireThreshold=args.acquire_threshold,
            trackThreshold=args.track_threshold,
        )

        sync = None

        def log_event(ev):
            row = {"monotonic": time.monotonic(), **ev}
            events.write(json.dumps(row) + "\n")
            events.flush()

            name = ev.get("event")
            if name in ("acquired", "reacquired", "track_miss", "spinner_mismatch"):
                fields = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("event", "state"))
                print(f"[{name.upper()}] state={ev.get('state')} {fields}")

        def on_frame(frame_index, _frame, status, meta):
            latest["frame"] = frame_index
            latest["status"] = status
            latest["corr"] = meta.get("corr")
            latest["err"] = meta.get("timing_error_samples")
            latest["pred"] = meta.get("predicted_frame")
            latest["phase"] = frame_index % args.spinner_size

            frames_csv.writerow([
                frame_index,
                status,
                sync.state,
                frame_index % args.spinner_size,
                meta.get("guard_pos", ""),
                meta.get("corr", ""),
                meta.get("timing_error_samples", ""),
                meta.get("observed_period", ""),
                meta.get("period_estimate", sync.periodEstimate),
                meta.get("predicted_frame", ""),
            ])
            frames_file.flush()

        sync = FixedGuardSyncV1(cfg=cfg, onFrame=on_frame, onEvent=log_event)

        src = subprocess.Popen(
            ["parec", f"--device={RX_SINK}.monitor", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=10"],
            stdout=subprocess.PIPE,
            bufsize=0,
        )

        print("RX SYNC CHANNEL PROBE RUNNING")
        print(f"path: Chrome/Telemost -> {RX_SINK} -> Python -> FixedGuardSyncV1")
        print("NOW start the TX probe and press ENTER on TX.")
        print("Press Ctrl+C to stop RX.")

        next_print = time.monotonic()
        total_samples = 0

        while True:
            data = read_exact(src.stdout, args.io_samples * 2)
            if len(data) != args.io_samples * 2:
                break

            x = pcm_to_float(data)
            raw_wav.write(x)
            total_samples += len(x)

            sync.push(x, tailWall=time.monotonic())

            now = time.monotonic()
            if now >= next_print:
                s = sync.stats
                print(
                    f"RX state={sync.state:9s} "
                    f"last={str(latest['frame']):>7s} "
                    f"phase={str(latest['phase']):>4s} "
                    f"corr={str(latest['corr']):>10s} "
                    f"err={str(latest['err']):>8s} "
                    f"pred={str(latest['pred']):>10s} "
                    f"ok={s['frames_ok']} lost={s['frames_lost']} "
                    f"reacq={s['reacquisitions']} mismatch={s['spinner_mismatches']}"
                )
                next_print = now + 1.0

    except KeyboardInterrupt:
        print("RX stopped")
    finally:
        if 'sync' in locals() and sync is not None:
            summary = {
                **sync.stats,
                "state_final": sync.state,
                "last_frame": sync.lastFrame,
                "period_estimate": sync.periodEstimate,
                "spinner_size": args.spinner_size,
            }
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(json.dumps(summary, indent=2))

        if raw_wav:
            raw_wav.close()
        if events:
            events.close()
        if frames_file:
            frames_file.close()
        if src:
            src.terminate()
        modules.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("tx", "rx"), required=True)
    p.add_argument("--out", required=True)

    p.add_argument("--spinner-size", type=int, default=8)
    p.add_argument("--guard-amp", type=float, default=0.12)

    p.add_argument("--acquire-threshold", type=float, default=0.20)
    p.add_argument("--track-threshold", type=float, default=0.14)

    p.add_argument("--io-samples", type=int, default=160)

    args = p.parse_args()

    if args.spinner_size != 8:
        raise ValueError("For this test keep --spinner-size 8")

    if not (0.0 < args.guard_amp < 1.0):
        raise ValueError("--guard-amp must be between 0 and 1")

    if args.io_samples <= 0:
        raise ValueError("--io-samples must be > 0")

    if args.role == "tx":
        run_tx(args)
    else:
        run_rx(args)


if __name__ == "__main__":
    main()
