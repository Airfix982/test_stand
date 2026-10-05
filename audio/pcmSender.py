import subprocess
import threading
import time


class GeneratedPcmSender:
    def __init__(self, sink, spec, source, duration, tap=None, logger=None):
        self.sink = sink
        self.spec = spec
        self.source = source
        self.duration_s = float(duration)
        self.tap = tap
        self.log = logger or (lambda *args, **kwargs: None)
        self.stopEvent = threading.Event()
        self.thread = None
        self.writer = None
        self.samplesOut = 0

    def _writerCmd(self):
        return ["pacat", "--playback", f"--device={self.sink}", "--format=s16le", f"--rate={self.spec.rate}", f"--channels={self.spec.channels}", f"--latency-msec={self.spec.latency}", f"--process-time-msec={self.spec.processTime}", "--raw"]

    def start(self):
        self.stopEvent.clear()
        self.writer = subprocess.Popen(self._writerCmd(), stdin=subprocess.PIPE, bufsize=0)
        self.thread = threading.Thread(target=self._loop, name="probe-sender", daemon=True)
        self.thread.start()
        self.log("probe_sender_started", sink=self.sink, duration_s=self.duration_s)

    def _loop(self):
        started = time.monotonic()
        block_index = 0
        try:
            while not self.stopEvent.is_set() and time.monotonic() - started < self.duration_s:
                samples = self.source.block(block_index)
                data = samples.tobytes()
                if self.tap:
                    self.tap.write(data)
                self.writer.stdin.write(data)
                self.writer.stdin.flush()
                self.samplesOut += len(samples)
                self.log("probe_block_sent", block_index=block_index, sample_start=self.samplesOut-len(samples), sample_count=len(samples))
                block_index += 1
        except (BrokenPipeError, OSError) as ex:
            if not self.stopEvent.is_set():
                self.log("probe_sender_error", error=repr(ex))
        finally:
            self.log("probe_sender_loop_ended", samples_out=self.samplesOut, blocks=block_index)

    def stop(self):
        self.stopEvent.set()
        if self.writer and self.writer.poll() is None:
            self.writer.terminate()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.writer and self.writer.poll() is None:
            self.writer.kill()
        self.writer = None
        if self.tap:
            self.tap.close()
        self.log("probe_sender_stopped", samples_out=self.samplesOut)