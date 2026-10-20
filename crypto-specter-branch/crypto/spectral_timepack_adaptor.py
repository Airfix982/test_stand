"""Universal FrameCryptoPlugin adapter for the UNMODIFIED git_core STFT/MDCT ciphers.

The spectral frame/hop belongs to the cipher; the 320-sample frame belongs to
TimePack. They are deliberately NOT treated as the same frame index.

Streaming discipline:
* TX delays transformed PCM by a constant whole number of TimePack frames.
* RX skips that TX startup latency, feeds the same spectral sample timeline,
  and adds its own decoder delay to cover spectral look-ahead.
* RX supplies silence for every lost TimePack slot. This preserves spectral
  hop numbering/key-schedule, but cannot restore discarded ciphertext samples.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from crypto.frame_port import FRAME, FrameCryptoPlugin, check_batch, check_frame
from spectral_core.crypto.stft import STFTCipher
from spectral_core.crypto.mdct import MDCTCipher


class SpectralTimepackAdaptor(FrameCryptoPlugin):
    name = 'spectral_signflip'
    encode_batch_size = 4

    def __init__(self, key_file, transform='stft', operation='signflip',
                 key_schedule='per_frame', session='stft-live-v1',
                 stft_frame=2048, stft_hop=1024, mdct_half=2500,
                 max_gap_frames=4096):
        # Key material passed directly to the ORIGINAL SpectralCipher.
        # DO NOT derive a new HMAC key here: base.py owns its original schedule.
        key = Path(key_file).expanduser().read_bytes()
        if not key:
            raise ValueError('Empty key file')
        if transform == 'stft':
            def build():
                return STFTCipher(operation=operation, key=key,
                                  session_nonce=session, key_schedule=key_schedule,
                                  frame_size=stft_frame, hop_size=stft_hop)
        elif transform == 'mdct':
            def build():
                return MDCTCipher(operation=operation, key=key,
                                  session_nonce=session, key_schedule=key_schedule,
                                  half_size=mdct_half)
        else:
            raise ValueError('transform must be stft or mdct')
        self.tx_cipher = build()
        self.rx_cipher = build()
        self.transform = transform
        self.operation = operation
        self.key_schedule = key_schedule
        self.frame_samples = self.tx_cipher.frame_size
        self.hop_samples = self.tx_cipher.hop_size
        # One whole-number amount of transport frames, at least the analysis
        # window. No variable-length callbacks or frame reordering.
        self.delay_frames = (self.frame_samples + self.hop_samples + FRAME - 1) // FRAME
        self.delay_samples = self.delay_frames * FRAME
        self._tx_queue = np.zeros(self.delay_samples, dtype=np.float64)
        # Identical to git_core/app.py: discard the first transform hop as
        # warmup (it has extremely large STFT edge amplitudes).
        self._tx_warmup_to_discard = self.hop_samples
        # First transmitted spectral block belongs to transform frame #1.
        self.rx_cipher.reset(frame_index=1)
        self._rx_warmup_to_mute = self.hop_samples
        self._rx_queue = np.zeros(self.delay_samples, dtype=np.float64)
        self._tx_next = 0
        self._rx_next = 0
        self.max_gap_frames = int(max_gap_frames)
        self.lost_input_frames = 0
        self.implicit_gap_frames = 0
        self.resets = 0
        # No buffer or key state is shared between directions.

    @staticmethod
    def _pop_append(queue, new, count):
        if len(new):
            queue = np.concatenate((queue, np.asarray(new, dtype=np.float64)))
        if len(queue) < count:
            raise RuntimeError('Spectral output underflow: latency bound too small')
        return queue[:count].copy(), queue[count:]

    def encode_batch(self, frames, indices):
        frames = check_batch(frames, indices)
        indices = [int(x) for x in indices]
        if indices != list(range(self._tx_next, self._tx_next + len(indices))):
            raise ValueError('TX must supply consecutive TimePack indices starting at zero')
        samples = frames.reshape(-1).astype(np.float64)
        encrypted = self.tx_cipher.encrypt(samples)
        if self._tx_warmup_to_discard:
            drop = min(self._tx_warmup_to_discard, len(encrypted))
            encrypted = encrypted[drop:]
            self._tx_warmup_to_discard -= drop
        out, self._tx_queue = self._pop_append(self._tx_queue, encrypted, len(samples))
        self._tx_next += len(indices)
        return out.reshape(len(indices), FRAME).astype(np.float32)

    def _rx_one(self, payload, index):
        # Initial TX transport frames are fixed digital silence, not part of
        # the spectral ciphertext time axis. Do not feed them into STFT/MDCT.
        if index < self.delay_frames:
            return np.zeros(FRAME, dtype=np.float32)
        reconstructed = self.rx_cipher.decrypt(np.asarray(payload, dtype=np.float64))
        # The first inverse-transform hop has the same nonzero-window edge
        # singularity as TX. Silence ONLY that startup hop (no key changes).
        if self._rx_warmup_to_mute and len(reconstructed):
            reconstructed = reconstructed.copy()
            mute = min(len(reconstructed), self._rx_warmup_to_mute)
            reconstructed[:mute] = 0.0
            self._rx_warmup_to_mute -= mute
        out, self._rx_queue = self._pop_append(self._rx_queue, reconstructed, FRAME)
        return out.astype(np.float32)

    def decode_frame(self, pcm, ctx):
        index = int(ctx.index)
        if index < 0:
            raise ValueError('Negative absolute index')
        if index < self._rx_next:
            # Ignore delayed/duplicate callbacks: never rewind running
            # spectral overlap state and silently use an unrelated key.
            return np.zeros(FRAME, dtype=np.float32)
        gap = index - self._rx_next
        if gap > self.max_gap_frames:
            raise RuntimeError('RX gap %d frames exceeds --max-gap-frames %d; cannot safely reconstruct stream phase' % (gap, self.max_gap_frames))
        for skipped in range(self._rx_next, index):
            self._rx_one(np.zeros(FRAME), skipped)
        self.implicit_gap_frames += gap
        self._rx_next = index + 1
        if ctx.status == 'ok' and pcm is not None:
            incoming = check_frame(pcm)
        else:
            incoming = np.zeros(FRAME, dtype=np.float32)
            self.lost_input_frames += 1
        return self._rx_one(incoming, index)

    def on_sync_event(self, event):
        # The synchronizer is authoritative for transport-frame boundaries.
        # It must NOT set the STFT frame counter to the 20ms frame number.
        pass

    def close(self):
        pass
