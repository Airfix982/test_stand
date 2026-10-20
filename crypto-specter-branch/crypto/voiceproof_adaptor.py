"""Diagnostic, reversible 20-ms PCM scramble for testing TimePack frame identity.

NOT cryptographically secure voice encryption. Audio cipher is intentionally
chosen for perceptual A/B tests across speech codecs, not confidentiality.

Wire payload: 320 float PCM samples @16k. No change to PN or TimePack.
Each frame has a separately derived, deterministic sample permutation (grain=1 by default) plus
independent polarity inversion based on master key, session and frame index.
"""
import hashlib
import hmac
from pathlib import Path

import numpy as np
from crypto.frame_port import FrameCryptoPlugin, FrameContext, FRAME, check_batch, check_frame


class VoiceProofAdaptor(FrameCryptoPlugin):
    name = 'voiceproof_hard_sample_scramble_test_NOT_SECURE'
    encode_batch_size = 4

    def __init__(self, key_file, session='voiceproof-1', grain=1, index_offset=0):
        self.key = Path(key_file).read_bytes()
        if len(self.key) < 16:
            raise ValueError('Use a shared binary key with at least 16 bytes')
        self.session = session.encode('utf-8')
        self.grain = int(grain)
        if FRAME % self.grain or self.grain <= 0 or self.grain >= FRAME:
            raise ValueError('grain must evenly divide 320 and be < 320 (recommended 1 for unambiguous speech test)')
        self.index_offset = int(index_offset)
        self._cache_index = None
        self._cache_plan = None

    def _plan(self, frame_index):
        index = int(frame_index) + self.index_offset
        if index < 0 or index >= 2**64:
            raise ValueError('frame index out of range')
        if index == self._cache_index:
            return self._cache_plan
        mac = hmac.new(self.key, b'NearS-VOICEPROOF-DIAG-1\x00' + len(self.session).to_bytes(2,'big') + self.session + index.to_bytes(8,'big'), hashlib.sha256).digest()
        rng = np.random.default_rng(int.from_bytes(mac[:16], 'big'))
        n = FRAME // self.grain
        plan = (rng.permutation(n), rng.integers(0, 2, size=n, dtype=np.uint8), rng.integers(0, 2, size=n, dtype=np.uint8))
        self._cache_index, self._cache_plan = index, plan
        return plan

    def _scramble(self, samples, frame_index):
        """Invertible per-frame sample permutation; optimized vectorized path."""
        x = check_frame(samples).reshape(-1, self.grain)
        perm, reverse, signs = self._plan(frame_index)
        y = x[perm].copy()
        if np.any(reverse) and self.grain > 1:
            y[reverse.astype(bool)] = y[reverse.astype(bool), ::-1]
        if np.any(signs):
            y[signs.astype(bool)] *= -1.0
        return y.reshape(FRAME)

    def _descramble(self, samples, frame_index):
        x = check_frame(samples).reshape(-1, self.grain)
        perm, reverse, signs = self._plan(frame_index)
        y = x.copy()
        if np.any(signs):
            y[signs.astype(bool)] *= -1.0
        if np.any(reverse) and self.grain > 1:
            y[reverse.astype(bool)] = y[reverse.astype(bool), ::-1]
        out = np.empty_like(y)
        out[perm] = y
        return out.reshape(FRAME)

    def encode_batch(self, frames, indices):
        x = check_batch(frames, indices)
        return np.stack([self._scramble(x[i], int(n)) for i,n in enumerate(indices)],axis=0).astype(np.float32)

    def decode_frame(self, pcm, ctx: FrameContext):
        if ctx.status != 'ok' or pcm is None:
            return np.zeros(FRAME, dtype=np.float32)
        return self._descramble(pcm,ctx.index)