"""Dedicated 320-sample cipher adapter (research/experimental)."""
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
    def encode(self,voice,index):
        y=self.m['encrypt_chunk'](pcm(voice).astype(np.int64),int(index),self.seed,META,320,1,'xor',3)
        return floats(y)
    def decode(self,wire,index):
        y=self.m['decrypt_chunk'](pcm(wire).astype(np.int64),int(index),self.seed,META,320,1,'xor',3)
        return floats(y)
    def close(self):pass
