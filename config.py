from dataclasses import asdict, dataclass

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
class AppConfig:
    channel: str = "telemost"
    recordsDir: str = "records"
    audio: AudioSpec = AudioSpec()
    devices: DeviceConfig = DeviceConfig()
    
    def toDict(self):
        return asdict(self)