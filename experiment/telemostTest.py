"""Telemost SuperSpinner 8x8 diagnostics: auto virtual audio AND forced Chrome routing.
Run from project root: python3 -m experiment.telemostSuperTest --role tx|rx
Requires analysis.telemostProbeV2 and experiment.syncV1Rx from the 8x8 release.
"""
import argparse
import os
import signal
import subprocess
import sys
import re
import threading
import time
from pathlib import Path

TX_SINK, TX_SOURCE, RX_SINK = "tx_enc_sink", "tx_mic", "rx_sink"


def pactl(*args):
    return subprocess.check_output(["pactl", *args], text=True).strip()


def matching_modules(role):
    tokens = ("source_name=tx_mic", "sink_name=tx_enc_sink") if role == "tx" else ("sink_name=rx_sink",)
    rows = []
    for line in pactl("list", "short", "modules").splitlines():
        cols = line.split(maxsplit=2)
        if len(cols) < 2:
            continue
        if any(token in line for token in tokens):
            rows.append((cols[0], line))
    return rows


def cleanup(role):
    rows = matching_modules(role)
    # Delete the remapped microphone before the virtual sink to which it points.
    if role == "tx":
        rows.sort(key=lambda row: "source_name=tx_mic" not in row[1])
    for module_id, line in rows:
        result = subprocess.run(["pactl", "unload-module", module_id], capture_output=True, text=True)
        if result.returncode:
            print(f"WARNING: could not unload module {module_id}: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"Removed old {role} module: {module_id}")


class VirtualAudio:
    def __init__(self, role):
        self.role = role
        self.owned = []

    def load(self, kind, *options):
        module_id = pactl("load-module", kind, *options)
        self.owned.append(module_id)
        return module_id

    def __enter__(self):
        cleanup(self.role)
        try:
            if self.role == "tx":
                self.load("module-null-sink", f"sink_name={TX_SINK}", "sink_properties=device.description=NearS_TX")
                self.load("module-remap-source", f"master={TX_SINK}.monitor", f"source_name={TX_SOURCE}", "source_properties=device.description=NearS_TX_Microphone")
                print("\nTX virtual microphone created: NearS_TX_Microphone (tx_mic)", flush=True)
            else:
                self.load("module-null-sink", f"sink_name={RX_SINK}", "sink_properties=device.description=NearS_RX")
                print("\nRX virtual speakers created: NearS_RX (rx_sink)", flush=True)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        for module_id in reversed(self.owned):
            subprocess.run(["pactl", "unload-module", module_id], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.owned.clear()
        print(f"\n{self.role.upper()} virtual devices cleaned up.", flush=True)



def _streams(kind):
    """Return PulseAudio streams without depending on output-localized labels elsewhere."""
    name = "source-outputs" if kind == "source" else "sink-inputs"
    header = r"^Source Output #(\d+)" if kind == "source" else r"^Sink Input #(\d+)"
    result = []
    for block in re.split(r"(?=^" + ("Source Output" if kind == "source" else "Sink Input") + r" #\d+)", pactl("list", name), flags=re.M):
        match = re.search(header, block, flags=re.M)
        if match and 'application.process.binary = "chrome"' in block:
            prop = "Source" if kind == "source" else "Sink"
            current = re.search(r"^\s*" + prop + r":\s*(\d+)\s*$", block, re.M)
            if current:
                result.append((match.group(1), current.group(1)))
    return result


def _index(kind, name):
    for row in pactl("list", "short", kind).splitlines():
        fields = row.split()
        if len(fields) >= 2 and fields[1] == name:
            return fields[0]
    return None


class AutoRouter:
    """Force the Chrome test call to our virtual devices; restore previous routes on exit."""
    def __init__(self, role):
        self.role = role
        self.stopEvent = threading.Event()
        self.old_routes = {}
        self.thread = None
        self.tx_ready = threading.Event()
        self.rx_ready = threading.Event()
        self.reported = set()

    def tick(self):
        if self.role == "tx":
            target = _index("sources", TX_SOURCE)
            if target is None:
                raise RuntimeError("tx_mic does not exist")
            for stream, current in _streams("source"):
                if current != target:
                    self.old_routes.setdefault(("source", stream), current)
                    subprocess.run(["pactl", "move-source-output", stream, TX_SOURCE], check=True, stdout=subprocess.DEVNULL)
                    print(f"\n[ROUTE TX] Chrome microphone #{stream}: source {current} -> {TX_SOURCE}", flush=True)
                else:
                    self.tx_ready.set()
            # Chrome must not inject received audio into the outgoing test signal.
            tx_sink_index = _index("sinks", TX_SINK)
            physical = next((line.split(":", 1)[1].strip() for line in pactl("info").splitlines() if line.startswith("Default Sink:")), None)
            if physical:
                for stream, current in _streams("sink"):
                    if current == tx_sink_index:
                        subprocess.run(["pactl", "move-sink-input", stream, physical], check=True, stdout=subprocess.DEVNULL)
                        print(f"\n[ROUTE TX] Chrome playback #{stream}: {TX_SINK} -> {physical}", flush=True)
            # Detect successful rerouting immediately, not only on next poll.
            if any(current == target for _, current in _streams("source")):
                self.tx_ready.set()
        else:
            target = _index("sinks", RX_SINK)
            if target is None:
                raise RuntimeError("rx_sink does not exist")
            for stream, current in _streams("sink"):
                if current != target:
                    self.old_routes.setdefault(("sink", stream), current)
                    subprocess.run(["pactl", "move-sink-input", stream, RX_SINK], check=True, stdout=subprocess.DEVNULL)
                    print(f"\n[ROUTE RX] Chrome playback #{stream}: sink {current} -> {RX_SINK}", flush=True)
                else:
                    self.rx_ready.set()
            if any(current == target for _, current in _streams("sink")):
                self.rx_ready.set()

    def _loop(self):
        while not self.stopEvent.wait(1.0):
            try:
                self.tick()
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                tag = str(exc)
                if tag not in self.reported:
                    self.reported.add(tag)
                    print(f"\n[ROUTE WARNING] {tag}", flush=True)

    def __enter__(self):
        self.tick()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def ensure_tx(self):
        if not self.tx_ready.wait(timeout=15):
            raise RuntimeError("Telemost/Chrome microphone stream not found! Join the call and leave Chrome microphone enabled, then retry.")
        print("[ROUTE TX] Browser microphone is connected to tx_mic.", flush=True)

    def __exit__(self, *_):
        self.stopEvent.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        for (kind, stream), previous in reversed(list(self.old_routes.items())):
            command = "move-source-output" if kind == "source" else "move-sink-input"
            subprocess.run(["pactl", command, stream, previous], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[ROUTE {self.role.upper()}] Previous Chrome stream routes restored (when still available).", flush=True)

def run_child(cmd):
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    env = dict(os.environ)
    # Prevent BLAS oversubscription when scanning 64 PN templates on a small machine.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    child = subprocess.Popen(cmd, start_new_session=True, env=env)
    try:
        return child.wait()
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping test and waiting for RX logs...", flush=True)
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
        try:
            return child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                return child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                return child.wait()


def main():
    p = argparse.ArgumentParser(description="Telemost 8x8 test: virtual devices automatically created, ENTER to start")
    p.add_argument("--role", choices=("tx", "rx"))
    p.add_argument("--cleanup-tx", action="store_true", help="Remove only old TX virtual devices and exit")
    p.add_argument("--cleanup-rx", action="store_true", help="Remove only old RX virtual devices and exit")
    p.add_argument("--duration", type=float, default=120)
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--out", help="Defaults: /tmp/ss_telemost_tx or /tmp/ss_telemost_rx")
    p.add_argument("--listen", action="store_true", help="RX: also monitor incoming test audio in physical headphones")
    args = p.parse_args()
    if args.cleanup_tx:
        cleanup("tx"); return
    if args.cleanup_rx:
        cleanup("rx"); return
    if not args.role:
        p.error("--role tx or --role rx is required")

    out = Path(args.out or f"/tmp/ss_telemost_{args.role}")
    if args.role == "tx":
        out.mkdir(parents=True, exist_ok=True)
        wav = out / "tx.wav"
        # Generate before devices are created, so nothing disappears while GUI settings are being changed.
        generator = [sys.executable, "-m", "analysis.telemostProbeV2", "generate", "--out", str(wav), "--duration", str(args.duration), "--seed", str(args.seed), "--super-spinner-size", "8"]
        print("Generating deterministic 8x8 TX signal...", flush=True)
        subprocess.run(generator, check=True)

    default_sink = next((line.split(":", 1)[1].strip() for line in pactl("info").splitlines() if line.startswith("Default Sink:")), None)
    with VirtualAudio(args.role) as devices:
        if args.role == "rx" and args.listen:
            if not default_sink:
                raise RuntimeError("Unable to find physical headphones for --listen")
            devices.load("module-loopback", f"source={RX_SINK}.monitor", f"sink={default_sink}", "latency_msec=150")
            print(f"[LISTEN] RX audio will ALSO be played into {default_sink}; use headphones to avoid feedback.", flush=True)
        if args.role == "tx":
            print("\nTELEMOST TX: select MICROPHONE 'NearS_TX_Microphone'.", flush=True)
            print("Enter only AFTER RX has started recording. Chrome microphone will be FORCE-ROUTED to tx_mic.", flush=True)
            input(f"PRESS ENTER to transmit the {args.duration:g}-second WAV... ")
            with AutoRouter("tx") as router:
                router.ensure_tx()
                print("TX sending... wait until paplay finishes.", flush=True)
                subprocess.run(["paplay", f"--device={TX_SINK}", str(wav)], check=True)
                print("TX finished. Stop RX after ~2 seconds.", flush=True)
        else:
            print("\nTELEMOST RX: select SPEAKERS 'NearS_RX'.", flush=True)
            print("Chrome playback will be FORCE-ROUTED to rx_sink; RX records it and does NOT play it in your headphones.", flush=True)
            input("PRESS ENTER to start the RX recorder (Ctrl+C to stop)... ")
            cmd = [sys.executable, "-m", "experiment.syncV1Rx", "--source", f"{RX_SINK}.monitor", "--out", str(out), "--super-spinner-size", "8", "--perf-interval", "10"]
            with AutoRouter("rx"):
                result = run_child(cmd)
            print(f"RX exit code: {result}", flush=True)
            if result != 0:
                raise SystemExit(result)
            print(f"\nRX WAV: {out / 'recovered.wav'}", flush=True)
            print(f"RX events: {out / 'events.jsonl'}", flush=True)
            print(f"RX frames: {out / 'frames.csv'}", flush=True)


if __name__ == "__main__":
    main()