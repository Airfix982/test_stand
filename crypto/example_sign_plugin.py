"""Example of custom plugin API. NOT SECURE ENCRYPTION."""
import numpy as np
from .frame_port import FrameCryptoPlugin, FRAME, check_batch, check_frame

class SignFlipDemo(FrameCryptoPlugin):
    name = 'example-sign-flip-NOT-SECURE'
    encode_batch_size = 1

    def __init__(self, keys, algo_dir=None, output_gain=1.0):
        self.keys = keys

    def _sign(self, frame_index):
        return -1.0 if int(self.keys.bits(frame_index,1)[0]) else 1.0

    def encode_batch(self, frames, indices):
        frames = check_batch(frames,indices)
        return np.stack([row * self._sign(int(n)) for row,n in zip(frames,indices)])

    def decode_frame(self, pcm, ctx):
        if pcm is None or ctx.status != 'ok':
            return np.zeros(FRAME,dtype=np.float32)
        return check_frame(pcm) * self._sign(ctx.index)
