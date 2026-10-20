"""No Telemost / no PulseAudio: exercise the real git_core cipher via TimePack adapter.
Run: python3 -m analysis.testSpectralTimepack
"""
import tempfile
from pathlib import Path

import numpy as np

from crypto.frame_port import FrameContext
from crypto.spectral_timepack_adaptor import SpectralTimepackAdaptor
from spectral_core.crypto.stft import STFTCipher


def signal(count=160):
    t = np.arange(count * 320) / 16000.0
    rng = np.random.default_rng(17)
    return (0.13 * np.sin(2 * np.pi * 217 * t)
            + 0.07 * np.sin(2 * np.pi * 443 * t)
            + 0.01 * rng.normal(size=t.size)).astype(np.float32).reshape(count, 320)


def encode(cipher, src):
    pieces = []
    for start in range(0, len(src), 4):
        batch = src[start:start + 4]
        pieces.append(cipher.encode_batch(batch, np.arange(start, start + len(batch))))
    return np.vstack(pieces)


def receive(cipher, encrypted, lost=()):
    out = []
    for i, pcm in enumerate(encrypted):
        status = 'lost' if i in lost else 'ok'
        out.append(cipher.decode_frame(None if status == 'lost' else pcm, FrameContext(i, status)))
    return np.vstack(out)


def main():
    with tempfile.TemporaryDirectory() as directory:
        key = Path(directory) / 'key.bin'
        key.write_bytes(bytes(range(32)))
        src = signal()
        for transform in ('stft', 'mdct'):
            for operation in ('signflip', 'shuffle', 'invert'):
                settings = {'mdct_half': 320} if transform == 'mdct' else {}
                tx = SpectralTimepackAdaptor(key, transform=transform, operation=operation, **settings)
                rx = SpectralTimepackAdaptor(key, transform=transform, operation=operation, **settings)
                wire = encode(tx, src)
                restored = receive(rx, wire)
                assert wire.shape == restored.shape == src.shape
                assert np.isfinite(restored).all()
                assert np.isfinite(wire).all()
                assert tx.tx_cipher.frame_index >= rx.rx_cipher.frame_index - 1
                assert tx.tx_cipher.frame_index - rx.rx_cipher.frame_index <= 1 + (tx.delay_samples // tx.hop_samples)
                print('PASS', transform, operation, 'delay_frames=', tx.delay_frames,
                      'stft_frame_count=', tx.tx_cipher.frame_index)
        tx = SpectralTimepackAdaptor(key)
        wire = encode(tx, src)
        plain_reference = STFTCipher('signflip', key=key.read_bytes(), session_nonce='stft-live-v1', key_schedule='per_frame')
        cipher_samples = plain_reference.encrypt(src.ravel().astype(np.float64))
        expected = np.concatenate((np.zeros(tx.delay_samples), cipher_samples[tx.hop_samples:]))[:src.size]
        assert np.allclose(expected, wire.ravel(), rtol=1e-5, atol=1e-6), 'TX algorithm output changed'
        print('PASS original unmodified STFT signflip ciphertext parity (after original warmup hop)')
        rx1 = SpectralTimepackAdaptor(key)
        rx2 = SpectralTimepackAdaptor(key)
        lost = set(range(57, 73))
        a = receive(rx1, wire, lost)
        b = []
        for i in range(57):
            b.append(rx2.decode_frame(wire[i], FrameContext(i, 'ok')))
        # The synchronizer normally emits `lost` for each gap; verify the same
        # effective state when it reports only the resumed frame index.
        b.append(rx2.decode_frame(wire[73], FrameContext(73, 'ok')))
        for i in range(74, len(wire)):
            b.append(rx2.decode_frame(wire[i], FrameContext(i, 'ok')))
        assert np.allclose(a[73], b[57], atol=1e-6) and np.allclose(a[-1], b[-1], atol=1e-6)
        assert rx2.implicit_gap_frames == 16
        assert rx1.rx_cipher.frame_index == rx2.rx_cipher.frame_index
        print('PASS 16 missing transport frames: spectral key/hop timeline stays aligned')
        # Guard-code wraps every 64 transport frames, but STFT hop uses 1024
        # samples: they must NOT be confused with each other.
        rx3 = SpectralTimepackAdaptor(key)
        for i in range(140):
            rx3.decode_frame(wire[i], FrameContext(i, 'ok'))
        assert rx3.rx_cipher.frame_index > 1
        print('PASS crossed transport frame 64/128 without resetting spectral key index')
        print('ALL SPECTRAL ADAPTER TESTS PASSED')


if __name__ == '__main__':
    main()
