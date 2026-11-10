"""ALG7 RX-only postprocessing helpers, Python 3.8+.

Keeps Encryption_class, RNG and TimePack unchanged.
Causal repair: unlike the original offline repair_decoded_indices(), this never
looks ahead, so there is no additional frame delay. OutputDynamics does not
raise silence aggressively and limits peaks before PCM16 conversion.
"""
from collections import deque
import numpy as np


class StreamingIndexRepair:
    def __init__(self, guard_low, guard_high, threshold=2500.0):
        self.lo = float(guard_low)
        self.hi = float(guard_high)
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        self._p = deque(maxlen=2)
        self._e = deque(maxlen=2)
        self.invalid_p = 0
        self.invalid_e = 0
        self.spike_p = 0
        self.spike_e = 0

    def _repair(self, v, hist, kind):
        v = float(v)
        invalid = not np.isfinite(v) or v < self.lo or v > self.hi
        if invalid:
            setattr(self, 'invalid_' + kind, getattr(self, 'invalid_' + kind) + 1)
            v = hist[-1] if hist else (self.lo + self.hi)/2.0
        v = float(np.clip(v, self.lo, self.hi))
        # Causal three-frame window: two already-repaired predecessors + current.
        # The original offline smoother uses previous/current/NEXT; causal != identical.
        if len(hist) == 2:
            med = float(np.median([hist[-2], hist[-1], v]))
            if abs(v - med) > self.threshold:
                setattr(self, 'spike_' + kind, getattr(self, 'spike_' + kind) + 1)
                v = med
        hist.append(v)
        return v

    def process(self, pitch_index, energy_index):
        return (self._repair(pitch_index, self._p, 'p'),
                self._repair(energy_index, self._e, 'e'))


class OutputDynamics:
    """20ms-frame AGC with fast reduction, slow recovery and peak limiter.

    Apply to UNCLIPPED float PCM after original synth output /32768.
    Keeping target_rms=None disables AGC, leaving just peak limitation.
    """
    def __init__(self, peak_ceiling=0.93, target_rms=None, max_boost=3.0,
                 floor_rms=0.003, release=0.04):
        self.ceiling = float(peak_ceiling)
        self.target_rms = None if target_rms is None else float(target_rms)
        self.max_boost = float(max_boost)
        self.floor_rms = float(floor_rms)
        self.release = float(release)
        self.agc = 1.0
        self.limiter = 1.0
        self.limited_frames = 0
        if not (0.0 < self.ceiling < 1.0):
            raise ValueError('peak_ceiling must be in (0,1)')

    def process(self, x):
        y = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0,
                          posinf=0.0, neginf=0.0).copy()
        rms = float(np.sqrt(np.mean(y*y)))
        if self.target_rms is not None:
            if rms > self.floor_rms:
                wanted = float(np.clip(self.target_rms / rms, 0.25, self.max_boost))
            else:
                # Avoid amplifying decoder background/silence aggressively.
                wanted = min(self.agc, 1.0)
            alpha = 0.45 if wanted < self.agc else 0.04
            self.agc += alpha*(wanted-self.agc)
            y *= self.agc
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        limit_target = min(1.0, self.ceiling / max(peak, 1e-12))
        if limit_target < self.limiter:
            self.limiter = limit_target  # immediate protection for transient peaks
        else:
            self.limiter += self.release*(limit_target-self.limiter)
        if self.limiter < .999999:
            self.limited_frames += 1
        y *= self.limiter
        return np.clip(y, -self.ceiling, self.ceiling).astype(np.float32)
