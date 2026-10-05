import wave
from pathlib import Path

from sources.probe import DeterministicProbe


def generateProbeWav(path, rate, durationSec, seed=4444, blockSec=1.0, levelDbfs=-15.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    probe = DeterministicProbe(rate, seed, blockSec, levelDbfs)
    totalSamples = int(round(durationSec * rate))
    written = 0
    block = 0

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)

        while written < totalSamples:
            x = probe.block(block)
            need = min(len(x), totalSamples - written)
            w.writeframes(x[:need].tobytes())
            written += need
            block += 1

    return path, written