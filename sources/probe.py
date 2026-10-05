import numpy as np


# измеритль сигнала

class DeterministicProbe:
    def __init__(self, rate, seed=4444, blockSeconds=1.0, levelDbfs=-15.0, lowHz=300.0, highHz=3400.0):
        self.rate = int(rate)
        self.seed = int(seed)
        self.blockSeconds = float(blockSeconds)
        self.blockSamples = int(round(self.rate * self.blockSeconds))
        self.levelDbfs = float(levelDbfs)
        self.lowHz = float(lowHz)
        self.highHz = min(float(highHz), self.rate * 0.45)




    def block(self, index):
        s = (self.seed + int(index) * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.default_rng(s)
        x = rng.standard_normal(self.blockSamples)
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(self.blockSamples, 1.0 / self.rate)
        X[(f < self.lowHz) | (f > self.highHz)] = 0
        x = np.fft.irfft(X, n=self.blockSamples)
        x -= np.mean(x)
        rms = np.sqrt(np.mean(x * x)) + 1e-12
        target = 32767.0 * (10.0 ** (self.levelDbfs / 20.0))
        x *= target / rms
        ramp_n = min(int(self.rate * 0.005), self.blockSamples // 4)
        if ramp_n > 1:
            ramp = np.linspace(0.0, 1.0, ramp_n, endpoint=False)
            x[:ramp_n] *= ramp
            x[-ramp_n:] *= ramp[::-1]
        return np.clip(np.rint(x), -32768, 32767).astype("<i2")