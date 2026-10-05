import csv
import subprocess
import threading
import time
from pathlib import Path


_FIELDS = ("rx_bytes", "rx_packets", "rx_errs", "rx_drop", "tx_bytes", "tx_packets", "tx_errs", "tx_drop")


def findDefaultInterface():
    p = subprocess.run(["ip", "route", "show", "default"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    parts = p.stdout.split()
    if "dev" in parts:
        i = parts.index("dev")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def readInterfaceStats(interface):
    for line in Path("/proc/net/dev").read_text().splitlines():
        if ":" not in line:
            continue
        name, values = line.split(":", 1)
        if name.strip() != interface:
            continue
        v = values.split()
        return dict(zip(_FIELDS, map(int, (v[0], v[1], v[2], v[3], v[8], v[9], v[10], v[11]))))
    raise RuntimeError(f"no network interface found: {interface}. ")


class NetworkStatsSampler:
    def __init__(self, path, interface=None, interval=1.0, logger=None):
        self.path = Path(path)
        self.interface = interface or findDefaultInterface()
        self.interval = float(interval)
        self.log = logger or (lambda *args, **kwargs: None)
        self.stopEvent = threading.Event()
        self.thread = None
        self._f = None
        self._writer = None

    def start(self):
        if not self.interface:
            self.log("network_stats_disabled", reason="default interface not detected")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["monotonic_ns", "interface", *_FIELDS])
        self._f.flush()
        self.stopEvent.clear()
        self.thread = threading.Thread(target=self._loop, name="network-stats", daemon=True)
        self.thread.start()
        self.log("network_stats_started", interface=self.interface)

    def _loop(self):
        while not self.stopEvent.is_set():
            try:
                s = readInterfaceStats(self.interface)
                self._writer.writerow([time.monotonic_ns(), self.interface, *(s[k] for k in _FIELDS)])
                self._f.flush()
            except Exception as exc:
                self.log("network_stats_error", error=repr(exc))
            self.stopEvent.wait(self.interval)

    def stop(self):
        self.stopEvent.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self._f:
            self._f.flush(); self._f.close(); self._f = None
        if self.interface:
            self.log("network_stats_stopped", interface=self.interface)