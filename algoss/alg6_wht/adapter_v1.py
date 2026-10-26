"""Standalone research cipher adapter: 320 PCM16 samples per frame.
Input/output are float32 [-1,1]; index is the SuperSpinner absolute frame number.
"""
import hashlib
from pathlib import Path
import numpy as np
from experiment.ALG6.ALG6_syncCipherSourceLoader import load_source
from experiment.ALG6.ALG6_syncCipherVdoRuntime import pcm, floats, N
META=dict(sample_rate=16000,sample_width_bytes=2,bits_per_sample=16,
          min_value=-32768,max_value=32767,dtype_name='int16')

def seed01(secret,index):
    d=hashlib.sha256(f'{secret}:{index}'.encode()).digest()
    return (int.from_bytes(d[:8],'big')+.5)/2**64

from functools import lru_cache
class Cipher:
    def __init__(self,secret):
        self.secret=secret;self.m=load_source(Path(__file__).with_name('wht_v3.py'))
        self.m['META_SECRET']=secret
    @lru_cache(maxsize=256)
    def positions(self,index):
        perm=self.m['build_permutation'](512,int(index))
        pos=np.flatnonzero(perm<128)
        assert len(pos)==128
        return pos
    def encode(self,voice,index):
        # 320->512 transform length; 128 kept int32 words -> 256 PCM16 words.
        x=np.pad(pcm(voice).astype(np.int64),(0,192))
        encrypted=self.m['encrypt_chunk'](x,META,int(index))
        words=np.asarray(encrypted,dtype='<i4')[self.positions(index)]
        return floats(np.pad(words.view('<i2'),(0,64)))
    def decode(self,wire,index):
        enc=np.frombuffer(pcm(wire)[:256].tobytes(),dtype='<i4')
        k1,k2=self.m['derive_key1_key2'](self.secret,int(index))
        key,_=self.m['build_key_norm_from_key1_key2'](k1,k2,512)
        base=self.m['float32_array_to_uint32_words'](key).copy()
        base[self.positions(index)]=enc.view('<u4')
        restored=self.m['decrypt_chunk'](base.view('<i4'),META,int(index))[:N]
        return floats(np.clip(restored,-32768,32767))
    def close(self):pass
