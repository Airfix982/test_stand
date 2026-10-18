"""Telemost SuperSpinner 8x8 diagnostics: automatic Pulse/PipeWire devices and ENTER gate.
Run from project root: python3 -m experiment.telemostSuperTest --role tx|rx
Requires analysis.telemostProbeV2 and experiment.syncV1Rx from the 8x8 release.
"""
import argparse
import signal
import subprocess
import sys
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


def run_child(cmd):
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    child = subprocess.Popen(cmd, start_new_session=True)
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

    with VirtualAudio(args.role):
        if args.role == "tx":
            print("\nTELEMOST TX: select MICROPHONE 'NearS_TX_Microphone'.", flush=True)
            print("Enter only AFTER RX has started recording.", flush=True)
            input(f"PRESS ENTER to transmit the {args.duration:g}-second WAV... ")
            print("TX sending... wait until paplay finishes.", flush=True)
            subprocess.run(["paplay", f"--device={TX_SINK}", str(wav)], check=True)
            print("TX finished. Stop RX after ~2 seconds.", flush=True)
        else:
            print("\nTELEMOST RX: select SPEAKERS 'NearS_RX'.", flush=True)
            print("Your physical speakers will NOT be selected: the file is recorded directly from the RX virtual sink.", flush=True)
            input("PRESS ENTER to start the RX recorder (Ctrl+C to stop)... ")
            cmd = [sys.executable, "-m", "experiment.syncV1Rx", "--source", f"{RX_SINK}.monitor", "--out", str(out), "--super-spinner-size", "8"]
            result = run_child(cmd)
            print(f"RX exit code: {result}", flush=True)
            if result != 0:
                raise SystemExit(result)
            print(f"\nRX WAV: {out / 'recovered.wav'}", flush=True)
            print(f"RX events: {out / 'events.jsonl'}", flush=True)
            print(f"RX frames: {out / 'frames.csv'}", flush=True)


if __name__ == "__main__":
    main()