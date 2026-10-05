from dataclasses import asdict, dataclass, field

@dataclass(frozen=True)
class AudioSpec:
    rate: int = 44100
    channels: int = 1
    sampleWidth: int = 2
    ioBlockSamples: int = 1024
    latency: int = 20
    processTime: int = 10

    @property
    def ioBlockBytes(self):
        return self.ioBlockSamples * self.channels * self.sampleWidth
    
@dataclass(frozen = True)
class DeviceConfig:
    micro: str = "alsa_input.pci-0000_04_00.6.analog-stereo"
    headphones: str = "alsa_output.pci-0000_04_00.6.analog-stereo"
    tx: str = "tx_hol"
    rx: str = "rx_hol"
    txSrc: str = "tx_micro"

    @property
    def rxSrc(self):
        return self.rx + ".monitor"
    
@dataclass(frozen=True)
class CheckerConfig:
    role: str = "full"
    mode: str = "live"
    durationSec: float = 60.0
    seed: int = 4444
    probeBlockSec: float = 1.0
    probeLvlDbfs: float = -15.0
    interface: str = ""


@dataclass(frozen=True)
class AppConfig:
    channel: str = "telemost"
    recordsDir: str = "records"
    audio: AudioSpec = field(default_factory=AudioSpec)
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    checker: CheckerConfig = field(default_factory=CheckerConfig)

    def toDict(self):
        return asdict(self)