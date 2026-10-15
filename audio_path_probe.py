import argparse
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np

RATE = 16000
BLOCK = 320

TX_SINK = "route_test_tx_sink"
TX_MIC = "route_test_tx_mic"
RX_SINK = "route_test_rx_sink"

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

    time.sleep(0.3)

    expected = index_for_name("sources", TX_MIC)
    rows = browser_streams("source-outputs")
    bad = False

    for row in rows:
        ok = row.get("route_index") == expected
        print(f"TX ROUTE {'OK' if ok else 'BAD'}: browser #{row['id']} source={row.get('route_index')} expected={expected} app={row.get('application')} media={row.get('media')}")
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

    time.sleep(0.3)

    expected = index_for_name("sinks", RX_SINK)
    rows = browser_streams("sink-inputs")
    bad = False

    for row in rows:
        ok = row.get("route_index") == expected
        print(f"RX ROUTE {'OK' if ok else 'BAD'}: browser #{row['id']} sink={row.get('route_index')} expected={expected} app={row.get('application')} media={row.get('media')}")
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


class Wav:
    def __init__(self, path):
        self.w = wave.open(str(path), "wb")
        self.w.setnchannels(1)
        self.w.setsampwidth(2)
        self.w.setframerate(RATE)

    def write(self, x):
        self.w.writeframes(float_to_pcm(x))

    def close(self):
        self.w.close()


def tx_worker(stop, physical_source, outdir, proof):
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

    w_in = Wav(Path(outdir) / "tx_physical.wav")
    w_out = Wav(Path(outdir) / "tx_to_virtual.wav")

    sample_index = 0
    energy_in = 0.0
    energy_out = 0.0
    count = 0
    next_print = time.monotonic() + 1.0

    try:
        while not stop.is_set():
            data = read_exact(src.stdout, BLOCK * 2)
            if len(data) != BLOCK * 2:
                break

            x = pcm_to_float(data)
            y = x.copy()

            if proof:
                n = np.arange(BLOCK, dtype=np.int64) + sample_index
                mask = (n % RATE) < int(0.120 * RATE)
                y[mask] += 0.12 * np.sin(2.0 * np.pi * 880.0 * n[mask] / RATE)

            y = np.clip(y, -0.999, 0.999)

            w_in.write(x)
            w_out.write(y)
            dst.stdin.write(float_to_pcm(y))

            energy_in += float(np.sum(x * x))
            energy_out += float(np.sum(y * y))
            count += len(x)
            sample_index += len(x)

            if time.monotonic() >= next_print:
                rms_in = (energy_in / max(count, 1)) ** 0.5
                rms_out = (energy_out / max(count, 1)) ** 0.5
                print(f"TX PATH: physical_rms={rms_in:.6f} python_to_virtual_rms={rms_out:.6f}", flush=True)
                energy_in = 0.0
                energy_out = 0.0
                count = 0
                next_print += 1.0
    finally:
        w_in.close()
        w_out.close()
        src.terminate()
        dst.terminate()


def rx_worker(stop, physical_sink, outdir):
    src = subprocess.Popen(
        ["parec", f"--device={RX_SINK}.monitor", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"],
        stdout=subprocess.PIPE,
        bufsize=0,
    )
    dst = subprocess.Popen(
        ["pacat", f"--device={physical_sink}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"],
        stdin=subprocess.PIPE,
        bufsize=0,
    )

    w_in = Wav(Path(outdir) / "rx_from_virtual.wav")
    w_out = Wav(Path(outdir) / "rx_to_physical.wav")

    energy = 0.0
    count = 0
    next_print = time.monotonic() + 1.0

    try:
        while not stop.is_set():
            data = read_exact(src.stdout, BLOCK * 2)
            if len(data) != BLOCK * 2:
                break

            x = pcm_to_float(data)
            w_in.write(x)
            w_out.write(x)
            dst.stdin.write(data)

            energy += float(np.sum(x * x))
            count += len(x)

            if time.monotonic() >= next_print:
                rms = (energy / max(count, 1)) ** 0.5
                state = "SIGNAL" if rms > 1e-6 else "SILENCE"
                print(f"RX PATH: {state} virtual_rms={rms:.6f} -> physical speaker", flush=True)
                energy = 0.0
                count = 0
                next_print += 1.0
    finally:
        w_in.close()
        w_out.close()
        src.terminate()
        dst.terminate()


def run_tx(args):
    physical_source = get_default("Default Source")
    print(f"TX physical microphone: {physical_source}")

    cleanup_named_modules((f"sink_name={TX_SINK}", f"source_name={TX_MIC}"))
    modules = Modules()
    stop = threading.Event()
    worker = None

    try:
        modules.load("module-null-sink", f"sink_name={TX_SINK}", "sink_properties=device.description=RouteTest_TX")
        modules.load("module-remap-source", f"master={TX_SINK}.monitor", f"source_name={TX_MIC}", "source_properties=device.description=RouteTest_TX_Microphone")

        print(f"TX virtual microphone ready: {TX_MIC}")
        print("TX IS PAUSED HERE. No Python audio worker is running yet.")
        input("TX COMPUTER: join the Telemost call in Chrome, then press ENTER here: ")

        worker = threading.Thread(target=tx_worker, args=(stop, physical_source, args.out, args.proof), daemon=True)
        worker.start()
        time.sleep(0.4)

        route_tx_browser()

        print("TX PATH ACTIVE")
        print(f"{physical_source} -> Python -> {TX_SINK} -> {TX_MIC} -> Chrome/Telemost")
        if args.proof:
            print("PROOF ON: the remote computer should hear a 120 ms / 880 Hz beep once per second.")
        print("Press Ctrl+C to stop TX.")

        while worker.is_alive():
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("TX stopping...")
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=2)
        modules.close()


def run_rx(args):
    physical_sink = get_default("Default Sink")
    print(f"RX physical speaker: {physical_sink}")

    cleanup_named_modules((f"sink_name={RX_SINK}",))
    modules = Modules()
    stop = threading.Event()
    worker = None

    try:
        modules.load("module-null-sink", f"sink_name={RX_SINK}", "sink_properties=device.description=RouteTest_RX")

        print(f"RX virtual speaker ready: {RX_SINK}")
        print("RX IS PAUSED HERE. No Python audio worker is running yet.")
        input("RX COMPUTER: join the Telemost call in Chrome, then press ENTER here: ")

        worker = threading.Thread(target=rx_worker, args=(stop, physical_sink, args.out), daemon=True)
        worker.start()
        time.sleep(0.4)

        route_rx_browser()

        print("RX PATH ACTIVE")
        print(f"Chrome/Telemost -> {RX_SINK} -> Python -> {physical_sink}")
        print("Press Ctrl+C to stop RX.")

        while worker.is_alive():
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("RX stopping...")
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=2)
        modules.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("tx", "rx"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--proof", action="store_true")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.role == "tx":
        run_tx(args)
    else:
        run_rx(args)


if __name__ == "__main__":
    main()