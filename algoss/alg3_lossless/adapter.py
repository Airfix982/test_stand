"""Dedicated 320-sample cipher adapter (research/experimental)."""
import hashlib
from pathlib import Path
import numpy as np
from experiment.ALG3.ALG3_syncCipherSourceLoader import load_source
from experiment.ALG3.ALG3_syncCipherServiceRuntime import pcm, floats, N
META=dict(sample_rate=16000,sample_width_bytes=2,bits_per_sample=16,
          min_value=-32768,max_value=32767,dtype_name='int16')
def seed01(secret,index):
    h=hashlib.sha256(f'{secret}:{index}'.encode()).digest()
    return (int.from_bytes(h[:8],'big')+.5)/2**64

class Cipher:
    def __init__(self,secret):
        self.secret=str(secret);self.m=load_source(Path(__file__).with_name('script_lossless.py'))
    def chaos(self,index):
        a,b,c=self.m['derive_session_params_from_secret'](self.secret+':'+str(int(index)))
        return self.m['ChebyshevSession'](a,b,c).generate(N)
    def encode(self,voice,index):
        y=self.m['encrypt_chunk'](floats(pcm(voice)).astype(np.float64),self.chaos(index))
        return (np.asarray(y)*2-1).astype(np.float32)
    def decode(self,wire,index):
        y=np.clip((floats(pcm(wire)).astype(np.float64)+1)/2,0,np.nextafter(1.,0.))
        return np.asarray(self.m['decrypt_chunk'](y,self.chaos(index)),np.float32)
    def close(self):pass
