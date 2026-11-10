"""Mock API contract test; original large ALG7 models/native encoder are NOT needed."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from unittest.mock import patch
from crypto import alg7_kra_adaptor as core
from crypto.frame_port import FrameContext


class Analyzer:
    speech_frame_size=160
    guard_low=100
    indices_range=1000
    pitch_period_min=40
    pitch_period_max=160
    energy_min=2
    energy_max=12
    def __init__(self,*args):pass
    def analyze_speech(self,x):
        assert x.shape==(640,)
        return (np.array([800,810,820,830],dtype=np.uint16),
                np.array([300,310,320,330],dtype=np.uint16),
                np.ones((8,4),np.float64)*.2,np.zeros(4),np.zeros(4),np.zeros((9,4)))

class Enc:
    encryption_bits=(8,16,24)
    quantization_levels=65536
    def __init__(self,*args):pass
    def Encrypt_translation(self,p,key):
        return (int(p)+sum(int(x)*(1<<i) for i,x in enumerate(key)))%self.quantization_levels
    def Decrypt_translation(self,p,key):
        return (int(p)-sum(int(x)*(1<<i) for i,x in enumerate(key)))%self.quantization_levels
    def Encrypt_rotation(self,t,key):return np.asarray(t)+.01*sum(key)
    def Decrypt_rotation(self,t,key):return np.asarray(t)-.01*sum(key)

class PS:
    Fs=16000; flng=240;glng=80;band_num=16;harm_num_max=3
    G_dim_half=8;G_dim=16;pitch_min=120;pitch_max=200
    # Intentionally 2D arrays: confirm production adapter reshapes them.
    c_0=np.ones((8,1));u_0=np.ones((16,1))
    filter_bank=np.zeros((16,120))
    def __init__(self,*args):pass
    def synthetize_frame(self,p,e,t,state):
        v=int(p)+int(e)
        return np.full(320,v,dtype=np.int16)
    def value_to_index(self,state):
        state.pitch_index=600
        state.energy_index=500
        state.timbre_index=np.ones(8)

class State:
    def __init__(self,*args):
        self.envelope_vector=np.zeros(16)
        self.timbre_value=np.zeros(16)
        self.energy_value=0
        self.pitch_value=130
        self.timbre_index=np.ones(8)

class Pitch:
    def __init__(self,*args):pass
    def est(self,*args,**kwargs):return .1

class Synth:
    def __init__(self,**kwargs):pass
    def synthesize_frame(self,*args,**kwargs):return np.ones(160)*1000


def main():
    with TemporaryDirectory() as temp:
        mods={'speech_analyzer':type('M',(),{'Speech_Analyzer_class':Analyzer}),
              'encryption':type('M',(),{'Encryption_class':Enc}),
              'pseudo_speech':type('M',(),{'Pseudospeech_Synthetizer_class':PS,
                                           'parameters_pseudospeech':State}),
              'single_pitch':type('M',(),{'single_pitch':Pitch}),
              'DirectLPCSynthesizer':type('M',(),{'DirectLPCSynthesizer':Synth})}
        with patch.object(core,'_load_module_tree',lambda directory:(Path(temp),mods)):
            a=core.Alg7KraAdaptor(temp,seed=37)
            assert a.ps.c_0.shape==(8,) and a.ps.u_0.shape==(16,)
            frames=np.zeros((4,320),np.float32)
            got=a.encode_batch(frames,np.arange(4,dtype=np.int64))
            assert got.shape==(4,320) and np.isfinite(got).all()
            out=a.decode_frame(got[0],FrameContext(0,'ok'))
            assert out.shape==(320,) and np.isfinite(out).all()
            lost=a.decode_frame(None,FrameContext(1,'lost'))
            assert (lost==0).all() and a.lost_count==1
            again=a.decode_frame(got[2],FrameContext(2,'ok'))
            assert again.shape==(320,)
            a.close()
    print('[PASS] ALG7 mock: 4-frame TX, PCM synthesis, RX, model radii normalization, key index, lost frame')

if __name__=='__main__':main()
