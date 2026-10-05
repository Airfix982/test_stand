import signal
import threading
import time

from audio.systemAudio import SystemWavPlayer, SystemWavRecorder
from audio.virtualDevices import VirtualDevicesManager
from experiment.session import ExperimentSession
from sources.probeFile import generateProbeWav
from stats.network import NetworkStatsSampler


class ChannelMeasurementRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stopEvent = threading.Event()
        self.session = ExperimentSession(cfg.recordsDir, cfg.channel, cfg.toDict())
        self.netstats = None

    def _log(self, event, **fields):
        self.session.log(event, **fields)

    def requestStop(self, *_):
        self.stopEvent.set()

    def run(self):
        signal.signal(signal.SIGINT, self.requestStop)
        signal.signal(signal.SIGTERM, self.requestStop)

        role = self.cfg.checker.role
        devs = self.cfg.devices
        audio = self.cfg.audio
        checker = self.cfg.checker
        base = self.session.path / "audio"

        if role == "tx":
            probePath = base / "probe_source.wav"
            probePath, samples = generateProbeWav(
                probePath, audio.rate, checker.durationSec,
                checker.seed, checker.probeBlockSec, checker.probeLvlDbfs
            )
            self._log("probe_generated", path=str(probePath), samples=samples, duration_samples=samples / audio.rate)

        try:
            with VirtualDevicesManager(devs, self._log):
                print("\nVirtual devices ready.")
                print("Configure Telemost now.")
                print("TX: microphone = TX_MICRO")
                print("RX: speakers   = RX")
                input(f"\nPress ENTER to start {role.upper()} measurement... ")

                self.netstats = NetworkStatsSampler(
                    self.session.path / "stats" / "network.csv",
                    checker.interface or None, 1.0, self._log
                )
                self.netstats.start()

                self._log("measurement_started", role=role)

                if role == "tx":
                    self._runTx(base, probePath)
                elif role == "rx":
                    self._runRx(base)
                else:
                    raise ValueError("measurement role must be tx or rx")

        finally:
            if self.netstats:
                self.netstats.stop()

            self._log("measurement_finished", role=role)

    def _runTx(self, base, probePath):
        audio = self.cfg.audio
        devs = self.cfg.devices

        # Пишем то, что реально существует на том SOURCE,
        # который читает Telemost.
        recorder = SystemWavRecorder(
            devs.txSrc, base / "tx_actual.wav",
            audio.rate, audio.channels, self._log
        )

        player = SystemWavPlayer(
            devs.tx, probePath, self._log
        )

        recorder.start()

        # Небольшой pre-roll, чтобы recorder гарантированно уже писал.
        if self.stopEvent.wait(0.5):
            recorder.stop()
            return

        player.start()

        try:
            while player.poll() is None:
                if self.stopEvent.wait(0.1):
                    player.stop()
                    break

            if player.poll() is not None:
                player.wait()

            # Небольшой хвост записи.
            self.stopEvent.wait(1.0)

        finally:
            recorder.stop()

    def _runRx(self, base):
        audio = self.cfg.audio
        devs = self.cfg.devices
        duration = self.cfg.checker.durationSec

        # Telemost пишет в RX sink.
        # Мы напрямую читаем monitor этого sink через parec.
        recorder = SystemWavRecorder(
            devs.rxSrc, base / "rx_channel.wav",
            audio.rate, audio.channels, self._log
        )

        recorder.start()

        try:
            started = time.monotonic()

            while not self.stopEvent.wait(0.2):
                elapsed = time.monotonic() - started

                if duration > 0 and elapsed >= duration:
                    break

                if int(elapsed) % 10 == 0:
                    pass

        finally:
            recorder.stop()