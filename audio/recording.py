import csv
import threading
import time
import wave
from pathlib import Path


class WavRecorder:
    def __init__(self, path, spec):
        self.path = Path(path)
        self.spec = spec
        self._wave = None
        self._lock = threading.Lock()
        self.samples = 0

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wave = wave.open(str(self.path), "wb")
        self._wave.setnchannels(self.spec.channels)
        self._wave.setsampwidth(self.spec.sampleWidth)
        self._wave.setframerate(self.spec.rate)
        return self

    def write(self, data):
        if not data or self._wave is None:
            return
        with self._lock:
            self._wave.writeframesraw(data)
            self.samples += len(data) // (self.spec.channels * self.spec.sampleWidth)

    def close(self):
        if self._wave is not None:
            with self._lock:
                self._wave.close()
                self._wave = None


class ChunkCsvLogger:
    def __init__(self, path, spec):
        self.path = Path(path)
        self.spec = spec
        self._f = None
        self._writer = None
        self._lock = threading.Lock()
        self.sample_index = 0

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["monotonic_ns", "sample_start", "sample_count", "byte_count"])
        self._f.flush()
        return self

    def write(self, data):
        samples = len(data) // (self.spec.channels * self.spec.sampleWidth)
        with self._lock:
            self._writer.writerow([time.monotonic_ns(), self.sample_index, samples, len(data)])
            self.sample_index += samples

    def close(self):
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None


class RecordingTap:
    def __init__(self, recorder=None, chunk_logger=None):
        self.recorder = recorder
        self.chunk_logger = chunk_logger

    def write(self, data):
        if self.recorder:
            self.recorder.write(data)
        if self.chunk_logger:
            self.chunk_logger.write(data)

    def close(self):
        if self.chunk_logger:
            self.chunk_logger.close()
        if self.recorder:
            self.recorder.close()