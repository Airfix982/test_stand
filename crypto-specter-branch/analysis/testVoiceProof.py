"""No-real-hardware regression: transport + Opus + wrong epoch + 8-frame erasure."""
import io
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
import numpy as np

from crypto.frame_port import FRAME,FrameContext
from crypto.voiceproof_adaptor import VoiceProofAdaptor
from sync.v2_timepack import SyncV1Config, FixedGuardSyncV1, makeGuardBank


def s16(y):return np.rint(np.clip(y,-1,1)*32767).astype('<i2')

def wavwrite(path,data,fs):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(fs);w.writeframes(s16(data).tobytes())

def wavread(path):
    with wave.open(str(path),'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768

def run_pcm(x,correct,wrong):
    received={};statuses={};events=[]
    def on_frame(index,payload,status,meta):
        received[index]=correct.decode_frame(np.asarray(payload)[:320] if status=='ok' else None,FrameContext(index,status))
        statuses[index]=status
    def on_event(ev):events.append(ev)
    cfg=SyncV1Config(rate=20000,frameSamples=400,coreSamples=320,guardSamples=80,spinnerSize=8,superSpinnerSize=8)
    rx=FixedGuardSyncV1(cfg,on_frame,on_event)
    for j in range(0,len(x),200):rx.push(x[j:j+200],None)
    rx.finish()
    return received,statuses,rx.stats

def main():
    with tempfile.TemporaryDirectory() as tmp:
        key=Path(tmp)/'key';key.write_bytes(bytes(range(32)))
        a=VoiceProofAdaptor(key,session='test',grain=80)
        b=VoiceProofAdaptor(key,session='test',grain=80)
        wrong=VoiceProofAdaptor(key,session='test',grain=80,index_offset=64)
        rng=np.random.default_rng(123)
        n=160
        t=np.arange(n*320)/16000
        # Harmonic, smoothly varying voiced speech-like carrier for Opus.
        phase=np.cumsum(2*np.pi*(140+60*np.sin(2*np.pi*0.3*t))/16000)
        signal=0.13*np.sin(phase)+0.045*np.sin(2*phase)+0.015*np.sin(3*phase)
        signal=(signal*(0.6+0.4*np.sin(2*np.pi*2*t)**2)).astype(np.float32).reshape(n,320)
        enc=a.encode_batch(signal,np.arange(n))
        perfect=np.stack([b.decode_frame(enc[i],FrameContext(i)) for i in range(n)])
        assert np.array_equal(signal,perfect), 'clean cipher is not exact inverse'
        w=np.stack([wrong.decode_frame(enc[i],FrameContext(i)) for i in range(n)])
        assert np.corrcoef(signal.ravel(),w.ravel())[0,1]<.25, 'wrong index unexpectedly decodes'
        print('PASS exact clean 320 PCM and wrong (+64) is different')
        bank=makeGuardBank(20000,80,64)
        wire=np.concatenate([np.concatenate((enc[i]*.75,bank[i%64]*.12)) for i in range(n)]).astype(np.float32)
        got,status,stats=run_pcm(wire,b,wrong)
        assert len(got)==n and all(s=='ok' for s in status.values()) and stats['frames_lost']==0
        print('PASS TimePack 8x8 clean, %d frames' % n)
        drop=np.concatenate((wire[:50*400],wire[(50+8)*400:]))
        got,status,stats=run_pcm(drop,b,wrong)
        assert len(got)==n and sum(s=='lost' for s in status.values())==8 and stats['frames_lost']==8 and max(got)==n-1
        print('PASS dropped 8 transport frames, no persistent key/epoch shift')
        if shutil.which('ffmpeg'):
            src=Path(tmp)/'wire.wav';opus=Path(tmp)/'wire.opus';out=Path(tmp)/'opus20.wav'
            wavwrite(src,wire,20000)
            subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(src),'-ar','48000','-ac','1','-c:a','libopus','-application','voip','-b:a','48k',str(opus)],check=True)
            subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(opus),'-ar','20000','-ac','1','-c:a','pcm_s16le',str(out)],check=True)
            got,status,stats=run_pcm(wavread(out),b,wrong)
            assert stats['frames_lost']==0 and len(got)==n
            good=np.concatenate([got[i] for i in range(n)])
            corr=np.corrcoef(signal.ravel(),good)[0,1]
            assert corr>.4, 'Opus destroyed decoded payload: %f' % corr
            print('PASS Opus 48k, all %d frames, correct Pearson %.3f' % (n,corr))
        else:print('SKIP Opus (ffmpeg missing)')
        print('ALL VOICEPROOF TESTS PASSED')

if __name__=='__main__':main()