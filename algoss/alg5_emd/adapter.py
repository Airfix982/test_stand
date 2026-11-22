"""Dedicated 320-sample cipher adapter (research/experimental)."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import numpy as np
from experiment.ALG5.ALG5_syncCipherSourceLoader import load_source
from experiment.ALG5.ALG5_syncCipherServiceRuntime import pcm, floats, N
META=dict(sample_rate=16000,sample_width_bytes=2,bits_per_sample=16,
          min_value=-32768,max_value=32767,dtype_name='int16')
def seed01(secret,index):
    h=hashlib.sha256(f'{secret}:{index}'.encode()).digest()
    return (int.from_bytes(h[:8],'big')+.5)/2**64

class Cipher:
    def __init__(self,secret):
        try:self.m=load_source(Path(__file__).with_name('emd_v3.py'))
        except ModuleNotFoundError as e:
            if e.name=='PyEMD':raise RuntimeError('Install EMD: python3 -m pip install EMD-signal') from e
            raise
        self.seed=int.from_bytes(hashlib.sha256(str(secret).encode()).digest()[:8],'big')
        x0, y0, self.r1, self.r2 = (self.m["derive_2dclm_params_from_chunk_index"](0, self.seed))
        x, y = self.m["generate_2dclm"](1, x0, y0, self.r1, self.r2, burn_in=1000)
        self._last_xy = (float(x[0]), float(y[0]))
        self._next_index = 0

        # Store only the last 256 frame starting states.
        self._cache = OrderedDict()
        self._cache_frames = 256

    def _state_for_frame(self, index):
        index = int(index)
        if index < 0:
            raise ValueError(f"Negative frame index: {index}")
        if index in self._cache:
            return self._cache[index]
        if index < self._next_index:
            raise RuntimeError(f"Frame {index} is older than retained {self._cache_frames}-frame window")

        if index - self._next_index > 20000:
            raise RuntimeError(f"Unexpected ALG5 frame jump: {index - self._next_index}")

        while self._next_index <= index:
            i = self._next_index
            self._cache[i] = self._last_xy
            x, y = self.m["generate_2dclm"](N,*self._last_xy,self.r1,self.r2,burn_in=0 if i == 0 else 1)
            self._last_xy = (float(x[-1]), float(y[-1]))
            self._next_index += 1

            if len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)

        return self._cache[index]


    def encode(self,voice,index):
        x0, y0 = self._state_for_frame(index)
        y=self.m['encrypt_chunk'](pcm(voice).astype(np.int64),int(index),self.seed,META,320,1,'xor',3,x0,y0,self.r1,self.r2)
        return floats(y)
    def decode(self,wire,index):
        x0, y0 = self._state_for_frame(index)
        y=self.m['decrypt_chunk'](pcm(wire).astype(np.int64),int(index),self.seed,META,320,1,'xor',3,x0,y0,self.r1,self.r2)
        return floats(y)
    def close(self):pass
