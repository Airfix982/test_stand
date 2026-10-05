import threading
import time
from experiment.session import ExperimentSession
from channel.profile import getChannelProfile
from pipeline.base import Pipeline
from algoss.identity import IdentityAlg
from sync.nosync import Nosync
from audio.pcmPipe import PcmPipe
from audio.virtualDevices import VirtualDevicesManager

import signal

from audio.pcmSender import GeneratedPcmSender
from audio.recording import ChunkCsvLogger, RecordingTap, WavRecorder
from sources.probe import DeterministicProbe
from stats.network import NetworkStatsSampler


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stopEvent = threading.Event()
        self.session = ExperimentSession(cfg.recordsDir, cfg.channel, cfg.toDict())
        self.channel = getChannelProfile(cfg.channel)
        self.txPipeline = Pipeline([IdentityAlg("tx"), Nosync("tx")])
        self.rxPipeline = Pipeline([Nosync("rx"), IdentityAlg("rx")])
        self.pipes = []
        self.sender = None
        self.netstats = None

    def requestStop(self, *_):
        self.stopEvent.set()

    def _log(self, event, **fields): self.session.log(event, **fields)

    def run(self):
        signal.signal(signal.SIGINT, self.requestStop)
        signal.signal(signal.SIGTERM, self.requestStop)
        devs = self.cfg.devices
        audio = self.cfg.audio
        checker = self.cfg.checker
        try:
            with VirtualDevicesManager(devs, self._log):
                self.netstats = NetworkStatsSampler(self.session.path / "stats" / "network.csv", checker.interface or None, 1.0, self._log); self.netstats.start()
                if(checker.role in ("rx", "full")):
                    self.pipes.append(PcmPipe("rx", devs.rxSrc, devs.headphones, audio, self.rxPipeline, inputTap=self._tap("rx_channel"), logger=self._log))
                if(checker.role in ("tx", "full")):
                    if(checker.mode=="probe"):
                        probe = DeterministicProbe(audio.rate, checker.seed, checker.probeBlockSec, checker.probeLvlDbfs)
                        self.sender = GeneratedPcmSender(devs.tx, audio, probe, checker.durationSec, self._tap("tx_exact"), self._log)
                    else:
                        self.pipes.append(PcmPipe("tx", devs.micro, devs.tx, audio, self.txPipeline, outputTap=self._tap("tx_exact"), logger=self._log))
                

                # self.pipes = [PcmPipe("tx", devs.micro, devs.tx, audio, self.txPipeline),
                #               PcmPipe("rx", devs.rxSrc, devs.headphones, audio, self.rxPipeline)];

                try:
                    input(f"enter to start")
                except (KeyboardInterrupt, EOFError):
                    self.stopEvent.set()
                    return

                if self.stopEvent.is_set():
                    return

                for pipe in self.pipes:
                    pipe.start()

                if self.sender:
                    self.sender.start()

                print("RUNNING")
                started = time.monotonic()


                nextStatus = started
                while not self.stopEvent.wait(.1):
                    now = time.monotonic()
                    if(self.sender and self.sender.thread and not self.sender.thread.is_alive()):
                        break
                    if(any(p.thread and not p.thread.is_alive() for p in self.pipes)):
                        raise RuntimeError("PCM pipe died for some reason. fock")
                    if(checker.durationSec > 0 and checker.mode != "probe" and now-started >= checker.durationSec):
                        break
                    if(now >= nextStatus):
                        vals = {p.name: p.bytesIn // (audio.channels * audio.sampleWidth) for p in self.pipes}
                        if(self.sender): 
                            vals["tx"] = self.sender.samplesOut
                        print(f"t={now-started:7.1f}s samples={vals}", flush=True); nextStatus = now + 5.0
        finally:
            if self.sender: self.sender.stop()
            for pipe in reversed(self.pipes):
                pipe.stop()
            if(self.netstats):
                self.netstats.stop()
            self.rxPipeline.close()
            self.txPipeline.close()

    def _tap(self, name):
        base = self.session.path / "audio"
        return RecordingTap(WavRecorder(base / f"{name}.wav", self.cfg.audio).start(), ChunkCsvLogger(base / f"{name}_chunks.csv", self.cfg.audio).start())
                    