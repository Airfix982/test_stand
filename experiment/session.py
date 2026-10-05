from pathlib import Path
from datetime import datetime
import threading
import json
import time

class ExperimentSession:
    def __init__(self, recordDir, channel, cfg):
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(recordDir) / f"{time}_{channel}"
        self.path.mkdir(parents=True,exist_ok=False)
        self.eventsPath = self.path / "events.jsonl"
        self._lock = threading.Lock()
        (self.path / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log("session_created", Path=str(self.path))

    def log(self, event, **kwargs):
        row = {"wall_time": datetime.now().isoformat(timespec="milliseconds"), "monotonic_ns": time.monotonic_ns(), "event": event, **kwargs}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with self.eventsPath.open("a", encoding="utf-8") as f:
                f.write(line + "\n")