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
    raise RuntimeError(f"{key} not found")


def cleanup_stale():
    text = pactl("list", "short", "modules")
    needles = (f"sink_name={TX_SINK}", f"source_name={TX_MIC}", f"sink_name={RX_SINK}")
    for line in reversed(text.splitlines()):
        if any(n in line for n in needles):
            subprocess.run(["pactl", "unload-module", line.split()[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Modules:
    def __init__(self):
        self.ids = []

    def load(self, *args):
        mid = pactl("load-module", *args)
        self.ids.append(mid)

    def close(self):
        for mid in reversed(self.ids):
            subprocess.run(["pactl", "unload-module", mid], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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


def move_and_verify_browser():
    capture = browser_streams("source-outputs")
    playback = browser_streams("sink-inputs")

    if not capture:
        raise RuntimeError("Browser microphone stream not found. Join the Telemost call first.")
    if not playback:
        raise RuntimeError("Browser playback stream not found. Join the Telemost call first.")

    for row in capture:
        subprocess.check_call(["pactl", "move-source-output", str(row["id"]), TX_MIC])

    for row in playback:
        subprocess.check_call(["pactl", "move-sink-input", str(row["id"]), RX_SINK])

    time.sleep(0.3)

    tx_index = index_for_name("sources", TX_MIC)
    rx_index = index_for_name("sinks", RX_SINK)

    capture = browser_streams("source-outputs")
    playback = browser_streams("sink-inputs")

    for row in capture:
        ok = row.get("route_index") == tx_index
        print(f"TX ROUTE {'OK' if ok else 'BAD'}: browser #{row['id']} app={row.get('application')} media={row.get('media')} source={row.get('route_index')} expected={tx_index}")
        if not ok:
            raise RuntimeError("TX browser stream did not move to virtual mic")

    for row in playback:
        ok = row.get("route_index") == rx_index
        print(f"RX ROUTE {'OK' if ok else 'BAD'}: browser #{row['id']} app={row.get('application')} media={row.get('media')} sink={row.get('route_index')} expected={rx_index}")
        if not ok:
            raise RuntimeError("RX browser stream did not move to virtual speaker")


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
    src = subprocess.Popen(["parec", f"--device={physical_source}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdout=subprocess.PIPE, bufsize=0)
    dst = subprocess.Popen(["pacat", f"--device={TX_SINK}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdin=subprocess.PIPE, bufsize=0)

    w_in = Wav(Path(outdir) / "tx_physical.wav")
    w_out = Wav(Path(outdir) / "tx_python_out.wav")

    sample_index = 0
    sum_in = 0.0
    sum_out = 0.0
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
                in_burst = (n % RATE) < int(0.120 * RATE)
                y[in_burst] += 0.12 * np.sin(2.0 * np.pi * 880.0 * n[in_burst] / RATE)

            y = np.clip(y, -0.999, 0.999)

            w_in.write(x)
            w_out.write(y)
            dst.stdin.write(float_to_pcm(y))

            sum_in += float(np.sum(x * x))
            sum_out += float(np.sum(y * y))
            count += len(x)
            sample_index += len(x)

            if time.monotonic() >= next_print:
                rms_in = (sum_in / max(count, 1)) ** 0.5
                rms_out = (sum_out / max(count, 1)) ** 0.5
                print(f"TX physical_rms={rms_in:.6f} python_out_rms={rms_out:.6f}")
                sum_in = 0.0
                sum_out = 0.0
                count = 0
                next_print += 1.0
    finally:
        w_in.close()
        w_out.close()
        src.terminate()
        dst.terminate()


def rx_worker(stop, physical_sink, outdir):
    src = subprocess.Popen(["parec", f"--device={RX_SINK}.monitor", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdout=subprocess.PIPE, bufsize=0)
    dst = subprocess.Popen(["pacat", f"--device={physical_sink}", "--format=s16le", f"--rate={RATE}", "--channels=1", "--latency-msec=20"], stdin=subprocess.PIPE, bufsize=0)

    w_in = Wav(Path(outdir) / "rx_virtual_in.wav")
    w_out = Wav(Path(outdir) / "rx_python_out.wav")

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
                print(f"RX {'SIGNAL' if rms > 1e-6 else 'SILENCE'} virtual_rms={rms:.6f} -> physical speaker")
                energy = 0.0
                count = 0
                next_print += 1.0
    finally:
        w_in.close()
        w_out.close()
        src.terminate()
        dst.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="route_probe")
    p.add_argument("--proof", action="store_true")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    physical_source = get_default("Default Source")
    physical_sink = get_default("Default Sink")

    cleanup_stale()
    modules = Modules()
    stop = threading.Event()

    try:
        modules.load("module-null-sink", f"sink_name={TX_SINK}", "sink_properties=device.description=RouteTest_TX")
        modules.load("module-remap-source", f"master={TX_SINK}.monitor", f"source_name={TX_MIC}", "source_properties=device.description=RouteTest_TX_Microphone")
        modules.load("module-null-sink", f"sink_name={RX_SINK}", "sink_properties=device.description=RouteTest_RX")

        print(f"Physical mic: {physical_source}")
        print(f"Physical speaker: {physical_sink}")
        print(f"Virtual TX mic: {TX_MIC}")
        print(f"Virtual RX speaker: {RX_SINK}")

        tx = threading.Thread(target=tx_worker, args=(stop, physical_source, outdir, args.proof), daemon=True)
        rx = threading.Thread(target=rx_worker, args=(stop, physical_sink, outdir), daemon=True)
        tx.start()
        rx.start()

        print("Join the Telemost call in Chrome now. The script will force the live Chrome streams onto the virtual devices.")
        input("When both microphone and playback streams exist, press ENTER: ")

        move_and_verify_browser()

        print("FULL DUPLEX ROUTE ACTIVE")
        print(f"TX: {physical_source} -> Python -> {TX_SINK} -> {TX_MIC} -> Chrome/Telemost")
        print(f"RX: Chrome/Telemost -> {RX_SINK} -> Python -> {physical_sink}")
        if args.proof:
            print("PROOF marker is ON: remote side should hear a short 880 Hz beep every second.")
        print("Ctrl+C to stop.")

        while tx.is_alive() and rx.is_alive():
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop.set()
        tx.join(timeout=2) if "tx" in locals() else None
        rx.join(timeout=2) if "rx" in locals() else None
        modules.close()


if __name__ == "__main__":
    main()
