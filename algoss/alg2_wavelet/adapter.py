"""Dedicated 320-sample cipher adapter (research/experimental)."""
import hashlib
from pathlib import Path
import numpy as np
from experiment.ALG2.ALG2_syncCipherSourceLoader import load_source
from experiment.ALG2.ALG2_syncCipherServiceRuntime import pcm, floats, N
META=dict(sample_rate=16000,sample_width_bytes=2,bits_per_sample=16,
          min_value=-32768,max_value=32767,dtype_name='int16')
def seed01(secret,index):
    h=hashlib.sha256(f'{secret}:{index}'.encode()).digest()
    return (int.from_bytes(h[:8],'big')+.5)/2**64

import math
class Cipher:
    def __init__(self,secret):
        self.m=load_source(Path(__file__).with_name('script_wavelet.py'))
        h=hashlib.sha512(str(secret).encode()).digest();mod=2**64+1
        s=[int.from_bytes(h[i:i+8],'big')%10**15+1 for i in (0,8,16)]
        self.keys=np.asarray(self.m['ChaosKeys'](16384,32768,mod,*s).generage_keys(),np.uint8)
    def key(self,index,n):
        off=(int(index)*640)%len(self.keys)
        return self.keys[(off+np.arange(n))%len(self.keys)]
    def encode(self,voice,index):
        # Haar/db1 APPROXIMATION only along time; 160x32-bit -> 320x16-bit
        x=pcm(voice).astype(np.int32)*65538
        c=np.trunc((x[::2].astype(np.float64)+x[1::2])/math.sqrt(2)).astype('<i4')
        b=c.view(np.uint8).copy();b^=self.key(index,len(b))
        return floats(b.view('<i2'))
    def decode(self,wire,index):
        b=pcm(wire).view(np.uint8).copy();b^=self.key(index,len(b))
        c=b.view('<i4').astype(np.float64)
        return np.clip(np.repeat(c/math.sqrt(2),2)/2147483647.,-1,1).astype(np.float32)
    def close(self):pass
