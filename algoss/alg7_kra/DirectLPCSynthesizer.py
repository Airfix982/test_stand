import numpy as np


class DirectLPCSynthesizer:
    def __init__(self, fs=8000, frame_size=160, lpc_order=12, periodic_mix=0.5, seed=12345):
        self.fs = int(fs)
        self.frame_size = int(frame_size)
        self.lpc_order = int(lpc_order)
        self.periodic_mix = float(periodic_mix)  # 0=noise, 1=pulse train

        self.history = np.zeros(self.lpc_order, dtype=np.float64)
        self.pitch_phase = 0.0
        self.deemph_mem = 0.0
        self.prev_pitch = None
        self.adaptive_mix = 0.5
        self.rng = np.random.default_rng(seed)

    def _estimate_periodic_mix(self, pitch_period, timbre):
        D = np.asarray(timbre, dtype=np.float64).reshape(-1)
        if len(D) != 9:
            raise ValueError(f"expected 9 timbre values, got {len(D)}")

        band_fraction = D * D
        total = np.sum(band_fraction)
        if total > 1e-12:
            band_fraction /= total

        low_energy = np.sum(band_fraction[:4])
        high_energy = np.sum(band_fraction[5:])
        tilt_db = 10.0 * np.log10((low_energy + 1e-9) / (high_energy + 1e-9))
        spectral_score = 1.0 / (1.0 + np.exp(-(tilt_db - 1.0) / 2.0))

        if self.prev_pitch is None:
            continuity_score = 0.5
        else:
            p0 = max(float(self.prev_pitch), 1e-6)
            p1 = max(float(pitch_period), 1e-6)
            jump_semitones = abs(12.0 * np.log2(p1 / p0))
            continuity_score = np.exp(-0.5 * (jump_semitones / 3.0) ** 2)

        self.prev_pitch = float(pitch_period)

        score = 0.75 * spectral_score + 0.25 * continuity_score
        target_mix = np.clip(0.05 + 0.90 * score, 0.05, 0.95)

        # Сглаживаем mix между кадрами, иначе сами создадим треск.
        self.adaptive_mix = 0.75 * self.adaptive_mix + 0.25 * target_mix
        return float(self.adaptive_mix)

    def _make_excitation(self, pitch_period, residual_energy, periodic_mix):
        N = self.frame_size
        pitch_period = max(float(pitch_period), 1.0)

        periodic = np.zeros(N, dtype=np.float64)
        for n in range(N):
            if self.pitch_phase <= 0.0:
                periodic[n] = 1.0
                self.pitch_phase += pitch_period
            self.pitch_phase -= 1.0

        noise = self.rng.standard_normal(N)

        pe = np.sum(periodic * periodic)
        if pe > 1e-12:
            periodic /= np.sqrt(pe)

        ne = np.sum(noise * noise)
        if ne > 1e-12:
            noise /= np.sqrt(ne)

        m = np.clip(periodic_mix, 0.0, 1.0)
        excitation = np.sqrt(m) * periodic + np.sqrt(1.0 - m) * noise

        ee = np.sum(excitation * excitation)
        residual_energy = max(float(residual_energy), 1e-12)

        if ee > 1e-12:
            excitation *= np.sqrt(residual_energy / ee)

        return excitation

    def _synthesis_filter(self, excitation, lpc):
        lpc = np.asarray(lpc, dtype=np.float64).reshape(-1)
        if len(lpc) != self.lpc_order:
            raise ValueError(f"expected {self.lpc_order} LPC coefficients, got {len(lpc)}")

        out = np.zeros(len(excitation), dtype=np.float64)
        hist = self.history.copy()

        for n in range(len(excitation)):
            prediction = -np.dot(lpc, hist)
            y = prediction + excitation[n]

            if not np.isfinite(y):
                y = 0.0

            y = np.clip(y, -1e9, 1e9)
            out[n] = y

            hist[1:] = hist[:-1]
            hist[0] = y

        self.history = hist
        return out

    def _deemphasis(self, signal):
        out = np.empty_like(signal, dtype=np.float64)
        prev = self.deemph_mem

        for n, x in enumerate(signal):
            y = x + 0.85 * prev
            out[n] = y
            prev = y

        self.deemph_mem = prev
        return out

    def synthesize_frame(self, pitch_period, log_prediction_error, lpc, timbre=None):
        residual_energy = 10.0 ** float(log_prediction_error)

        if timbre is None:
            mix = self.periodic_mix
        else:
            mix = self._estimate_periodic_mix(pitch_period, timbre)

        excitation = self._make_excitation(pitch_period, residual_energy, mix)
        signal = self._synthesis_filter(excitation, lpc)
        return self._deemphasis(signal)