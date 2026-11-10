import numpy as np

class HarmonicSpeechSynthesizer:
    def __init__(self, fs=8000, frame_size=160):
        self.fs = fs
        self.frame_size = frame_size
        self.phase = 0.0

    def synthesize_frame(self, pitch_period, energy, timbre):
        pitch_period = max(float(pitch_period), 1.0)

        f0 = self.fs / pitch_period
        nyquist = self.fs / 2.0

        harmonic_freqs = np.arange(f0, nyquist, f0, dtype=np.float64)

        if len(harmonic_freqs) == 0:
            return np.zeros(self.frame_size, dtype=np.int16)

        # timbre - 9-компонентная нормированная спектральная форма. Для НЕнейросетевогоо варианта считаем ее амплитудной огибающей 9 частотных полос
        envelope = np.maximum(np.asarray(timbre), 0.0)

        band_freqs = np.linspace(0.0, nyquist, len(envelope))

        harmonic_amp = np.interp(harmonic_freqs, band_freqs, envelope)

        n = np.arange(self.frame_size, dtype=np.float64)

        signal = np.zeros(self.frame_size, dtype=np.float64)

        for k, (freq, amp) in enumerate(zip(harmonic_freqs, harmonic_amp), start=1):
            signal += amp * np.cos(2.0 * np.pi * freq * n / self.fs + k * self.phase)

        current_energy = np.sum(signal * signal)

        if current_energy > 1e-12:
            signal *= np.sqrt(float(energy) / current_energy)

        self.phase += (2.0 * np.pi * f0 * self.frame_size / self.fs)
        self.phase %= 2.0 * np.pi

        return np.clip(signal, -32768, 32767).astype(np.int16)