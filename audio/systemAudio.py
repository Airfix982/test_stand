import signal
import subprocess
from pathlib import Path


class SystemWavPlayer:
    def __init__(self, sink, wavPath, logger=None):
        self.sink = sink
        self.wavPath = Path(wavPath)
        self.log = logger or (lambda *args, **kwargs: None)
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            ["paplay", f"--device={self.sink}", str(self.wavPath)],
            stdin=subprocess.DEVNULL
        )
        self.log("system_playback_started", sink=self.sink, path=str(self.wavPath), pid=self.proc.pid)

    def poll(self):
        return self.proc.poll() if self.proc else None

    def wait(self):
        code = self.proc.wait()
        self.log("system_playback_finished", returncode=code)
        if code != 0:
            raise RuntimeError(f"paplay exited with code {code}")

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()


class SystemWavRecorder:
    def __init__(self, source, wavPath, rate, channels=1, logger=None):
        self.source = source
        self.wavPath = Path(wavPath)
        self.rate = rate
        self.channels = channels
        self.log = logger or (lambda *args, **kwargs: None)
        self.proc = None

    def start(self):
        self.wavPath.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "parec", f"--device={self.source}", "--format=s16le",
            f"--rate={self.rate}", f"--channels={self.channels}",
            "--file-format=wav", str(self.wavPath)
        ]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        self.log("system_recording_started", source=self.source, path=str(self.wavPath), pid=self.proc.pid)

    def stop(self):
        if not self.proc or self.proc.poll() is not None:
            return

        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()

        self.log("system_recording_stopped", source=self.source, path=str(self.wavPath))