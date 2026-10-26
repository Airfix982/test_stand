"""Standalone research cipher adapter: 320 PCM16 samples per frame.
Input/output are float32 [-1,1]; index is the SuperSpinner absolute frame number.
"""
import hashlib
from pathlib import Path
import numpy as np
from experiment.ALG1.ALG1_syncCipherSourceLoader import load_source
from experiment.ALG1.ALG1_syncCipherVdoRuntime import pcm, floats, N
META=dict(sample_rate=16000,sample_width_bytes=2,bits_per_sample=16,
          min_value=-32768,max_value=32767,dtype_name='int16')

def seed01(secret,index):
    d=hashlib.sha256(f'{secret}:{index}'.encode()).digest()
    return (int.from_bytes(d[:8],'big')+.5)/2**64

class Cipher:
    def __init__(self, secret):
        self.secret=secret;self.m=load_source(Path(__file__).with_name('script1.py'))
    def encode(self,voice,index):
        x=pcm(voice).astype(np.int64)
        y,_=self.m['encrypt_signal'](x,META,seed01(self.secret,index),.9876543210987654,N)
        return floats(y)
    def decode(self,wire,index):
        y,_=self.m['decrypt_signal'](pcm(wire).astype(np.int64),META,seed01(self.secret,index),.9876543210987654,N)
        return floats(y)
    def close(self):pass
