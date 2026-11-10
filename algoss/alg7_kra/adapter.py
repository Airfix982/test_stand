"""ALG7/KRA -> the same Cipher API as algorithms 1–6.

Original encryption.py and np.random.default_rng(seed) are NOT modified.
The 4x20ms speech encoder, 16k pseudovoice synthesis and 8k LPC decoder
are owned by the earlier, audited crypto.alg7_kra_adaptor.
"""
from pathlib import Path
import numpy as np
from crypto.alg7_kra_adaptor import Alg7KraAdaptor, OriginalNumpyRngCursor
from crypto.frame_port import FrameContext


class Cipher:
    encode_batch_size = 4

    def __init__(self, secret='37', algo_dir=None, synth_gain=1.0):
        try:
            seed = int(secret)
        except (ValueError, TypeError):
            raise ValueError('ALG7 preserves original numpy RNG seed: use --secret 37 or --alg7-seed 37')
        directory = Path(algo_dir or Path(__file__).resolve().parent).resolve()
        self.impl = Alg7KraAdaptor(directory, seed=seed, output_gain=float(synth_gain))
        self.encode_batch_size = self.impl.encode_batch_size

    def encode_batch(self, frames, indices):
        return self.impl.encode_batch(frames, indices)

    def decode_frame(self, wire, ctx):
        # Shared runtime passes integer index (and a status for missing frames).
        if isinstance(ctx, FrameContext):
            context = ctx
        else:
            context = FrameContext(int(ctx), 'ok')
        return self.impl.decode_frame(wire, context)

    def decode_lost(self, index):
        return self.impl.decode_frame(None, FrameContext(int(index), 'lost'))

    def on_sync_event(self, event):
        self.impl.on_sync_event(event)

    def close(self):
        self.impl.close()


__all__ = ['Cipher', 'OriginalNumpyRngCursor']
