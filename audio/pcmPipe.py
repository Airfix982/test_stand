import threading
import subprocess

class PcmPipe:
    def __init__(self, name, source, sink, spec, processor, inputTap=None, outputTap=None, logger=None):
        self.name = name; 
        self.source = source; 
        self.sink = sink; 
        self.spec = spec; 
        self.processor = processor
        self.inputTap = inputTap; 
        self.outputTap = outputTap
        self.log = logger or (lambda *args, **kwargs: None)
        self.stopEvent = threading.Event()
        self.thread = None; 
        self.reader = None; 
        self.writer = None
        self.bytesIn = 0
        self.bytesOut = 0

    def start(self):
        if(self.thread and self.thread.is_alive()):
            return
        self.stopEvent.clear()
        self.reader = subprocess.Popen(self._readerCmd(), stdout=subprocess.PIPE, bufsize=0)
        self.writer = subprocess.Popen(self._writerCmd(), stdin=subprocess.PIPE, bufsize=0)
        self.thread = threading.Thread(target=self._loop, name=f"pcm-{self.name}", daemon=True)
        self.thread.start()
        self.log("pcm_pipe_started", name=self.name, source=self.source, sink=self.sink)

    def _readerCmd(self):
        return ["parec", f"--device={self.source}", "--format=s16le", f"--rate={self.spec.rate}", f"--channels={self.spec.channels}", f"--latency-msec={self.spec.latency}", f"--process-time-msec={self.spec.processTime}", "--raw"]

    def _writerCmd(self):
        return ["pacat", "--playback", f"--device={self.sink}", "--format=s16le", f"--rate={self.spec.rate}", f"--channels={self.spec.channels}", f"--latency-msec={self.spec.latency}", f"--process-time-msec={self.spec.processTime}", "--raw"]

    def _loop(self):
        try:
            while(not self.stopEvent.is_set()):
                data = self.reader.stdout.read(self.spec.ioBlockBytes)
                if(not data):
                    break;
                self.bytesIn += len(data)
                if self.inputTap: self.inputTap.write(data)
                out = self.processor.process(data)
                if(out):
                    if(self.outputTap): 
                        self.outputTap.write(out)
                    self.writer.stdin.write(out)
                    self.bytesOut += len(out)
        except(BrokenPipeError, OSError) as ex:
            if (not self.stopEvent.is_set()):
                self.log("pcm_pipe_error", name=self.name, error=repr(ex))
                print("PCM PIPE ERROR")
        finally:
            self.log("pcm_pipe_loop_ended", name=self.name, bytes_in=self.bytesIn, bytes_out=self.bytesOut)
            print("PCM PIPE LOOP ENDED")

    def stop(self):
        self.stopEvent.set()
        for proc in (self.reader, self.writer):
            if(proc and proc.poll() is None):
                proc.terminate()
        if(self.thread):
            self.thread.join(timeout=2.)
        for proc in (self.reader, self.writer):
            if(proc and proc.poll() is None):
                proc.kill()
        self.reader = None
        self.writer = None
        for tap in (self.outputTap, self.inputTap):
            if tap: 
                tap.close()
        