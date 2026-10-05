import threading
from experiment.session import ExperimentSession
from channel.profile import getChannelProfile
from pipeline.base import Pipeline
from algoss.identity import IdentityAlg
from sync.nosync import Nosync
from audio.pcmPipe import PcmPipe
from audio.virtualDevices import VirtualDevicesManager

import signal


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stopEvent = threading.Event()
        self.session = ExperimentSession(cfg.recordsDir, cfg.channel, cfg.toDict())
        self.channel = getChannelProfile(cfg.channel)
        self.txPipeline = Pipeline([IdentityAlg("tx"), Nosync("tx")])
        self.rxPipeline = Pipeline([Nosync("rx"), IdentityAlg("rx")])
        self.pipes = []

    def requestStop(self):
        self.stopEvent.set()

    def run(self):
        signal.signal(signal.SIGINT, self.requestStop)
        signal.signal(signal.SIGTERM, self.requestStop)
        devs = self.cfg.devices
        audio = self.cfg.audio
        try:
            with VirtualDevicesManager(devs):
                self.pipes = [PcmPipe("tx", devs.micro, devs.tx, audio, self.txPipeline),
                              PcmPipe("rx", devs.rxSrc, devs.headphones, audio, self.rxPipeline)];
                for pipe in self.pipes:
                    pipe.start()
                print("V0 RUNNING")
                while not self.stopEvent.wait(.5):
                    if(any(p.thread and not p.thread.is_alive() for p in self.pipes)):
                        raise RuntimeError("PCM pipe died for some reason. fock")
        finally:
            for pipe in reversed(self.pipes):
                pipe.stop()
            self.rxPipeline.close()
            self.txPipeline.close()
                    