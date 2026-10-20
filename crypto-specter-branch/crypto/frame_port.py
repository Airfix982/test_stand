"""NearS protocol-agnostic cipher adapter API. TimePack boundary: 320 PCM @ 16 kHz.
No prescribed encryption scheme, key derivation, or randomness source.
"""
from dataclasses import dataclass
import hashlib
import hmac
import numpy as np

RATE = 16000
FRAME = 320

@dataclass(frozen=True)
class FrameContext:
    index: int
    status: str = 'ok'

class FrameCryptoPlugin:
    """Implement encode_batch() and decode_frame(). Stateful synthesis is allowed.

    TX: encode_batch(pcm[N,320], indices[N]) -> encrypted_pcm[N,320].
    RX: decode_frame(encrypted_pcm[320], FrameContext) -> clear_pcm[320].
    On lost frame: decode_frame(None, FrameContext(index,'lost')) -> concealment PCM.
    Key policy is PRIVATE to each algorithm. A plugin may use a sequential RNG,
    a counter-based key, an external state machine, or no key at all.
    The transport supplies absolute frame indices but never derives keys.
    """
    name = 'abstract'
    encode_batch_size = 1

    def encode_batch(self, frames, indices):
        raise NotImplementedError

    def decode_frame(self, pcm, ctx):
        raise NotImplementedError

    def on_sync_event(self, event):
        """Optional sync notification. Algorithms own their state and key policy."""
        pass

    def close(self):
        pass

class FrameKeySource:
    """Random access HMAC-SHA256-derived per-frame bits; no seek, no RNG drift.

    This is a deterministic key derivation mechanism; it does NOT establish the
    security of an individual speech-encryption algorithm.
    """
    def __init__(self, master_key, session_id='demo-session'):
        if not isinstance(master_key, bytes) or len(master_key) < 16:
            raise ValueError('master_key must be at least 16 random bytes')
        self._key = master_key
        self._session = str(session_id).encode('utf-8')

    def bits(self, frame_index, count):
        if frame_index < 0 or frame_index >= 2**64:
            raise ValueError('frame_index out of range')
        if count < 0:
            raise ValueError('count out of range')
        prefix = b'NearS-crypto-frame-v1\x00' + len(self._session).to_bytes(2,'big') + self._session + int(frame_index).to_bytes(8,'big')
        blocks = []
        for block in range((count + 255) // 256):
            blocks.append(hmac.new(self._key, prefix + block.to_bytes(4,'big'), hashlib.sha256).digest())
        raw = b''.join(blocks)
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:count].copy()

    @classmethod
    def from_file(cls, filename, session_id='demo-session'):
        from pathlib import Path
        key_hex = Path(filename).read_text().strip()
        return cls(bytes.fromhex(key_hex), session_id)

    @classmethod
    def for_demo_only(cls, seed, session_id='demo-session'):
        key = hashlib.sha256(('INSECURE-demo-seed-' + str(seed)).encode()).digest()
        return cls(key, session_id)

class IdentityPlugin(FrameCryptoPlugin):
    name = 'identity'

    def encode_batch(self, frames, indices):
        return check_batch(frames, indices)

    def decode_frame(self, pcm, ctx):
        return np.zeros(FRAME, np.float32) if ctx.status != 'ok' or pcm is None else check_frame(pcm)


def check_frame(pcm):
    pcm = np.asarray(pcm, dtype=np.float32)
    if pcm.shape != (FRAME,):
        raise ValueError('Expected mono 320 float32 samples at 16kHz, got %r' % (pcm.shape,))
    if not np.all(np.isfinite(pcm)):
        raise ValueError('non-finite PCM')
    return pcm


def check_batch(frames, indices):
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != FRAME or len(frames) != len(indices):
        raise ValueError('Expected frames (N,320) and N absolute indices')
    if not np.all(np.isfinite(frames)):
        raise ValueError('non-finite PCM')
    return frames
