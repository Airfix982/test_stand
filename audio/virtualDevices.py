import subprocess
import shutil
import time

class VirtualDevicesManager:
    def __init__(self, config, logger=None):
        self.config = config
        self.log = logger or (lambda *args, **kwargs: None)
        self.deviceIds = []

    def _run(self, args, check=True):
        return subprocess.run(args, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def checkPactl(self):
        for tool in ["pactl", "parec", "pacat", "paplay"]:
            if not shutil.which(tool):
                raise RuntimeError("NO TOOL: " + tool)

    def removeOldVirtualDevices(self):
        devices = self._run(["pactl", "list", "short", "modules"])
        oldDevices = (self.config.tx, self.config.rx, self.config.txSrc)
        ids = []
        for line in devices.stdout.splitlines():
            if(any(dev in line for dev in oldDevices)):
                try:
                    ids.append(int(line.split()[0]))
                except:
                    pass
        for id in reversed(ids):
            self._run(["pactl", "unload-module", str(id)], check=False)
            self.log("stale_pulse_module_unloaded", module_id=id)

    def _load(self, module, *args):
        res = self._run(["pactl", "load-module", module, *args])
        id = int(res.stdout.strip())
        self.deviceIds.append(id)
        self.log("pulse_module_loaded", module=module, module_id=id, args=list(args))
        return id
    
    def _exist(self, name, type):
        res = self._run(["pactl", "list", "short", type], check=False)
        return any(len(parts := line.split()) > 1 and parts[1] == name for line in res.stdout.splitlines())
    
    def _verify(self):
        end = time.monotonic() + 2.
        while time.monotonic() < end:
            if self._exist(self.config.tx, "sinks") and self._exist(self.config.rx, "sinks") and self._exist(self.config.txSrc, "sources"):
                return
            time.sleep(.05)
        raise RuntimeError("DEVICES CREATED BUT DEVICES NOT FOUND")

    def start(self):
        self.checkPactl()
        self.removeOldVirtualDevices()
        try:
            self._load("module-null-sink", f"sink_name={self.config.tx}", "sink_properties=device.description=TX")
            self._load("module-null-sink", f"sink_name={self.config.rx}", "sink_properties=device.description=RX")
            self._load("module-remap-source", f"master={self.config.tx}.monitor", f"source_name={self.config.txSrc}", "source_properties=device.description=TX_MICRO")
            self._verify()
            print("virtual_audio_ready")
            self.log("virtual_audio_ready", tx=self.config.tx, rx=self.config.rx, txSrc=self.config.txSrc)
        except Exception:
            self.stop()
            raise

    def stop(self):
        for id in reversed(self.deviceIds):
            self._run(["pactl", "unload-module", str(id)], check=False)
            self.log("pulse_module_unloaded", module_id=id)
        self.deviceIds.clear()

    def __enter__(self):
        self.start()
        return self;

    def __exit__(self, excType, excValue, traceback):
        self.stop()
        