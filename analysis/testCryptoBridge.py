"""Offline TX/RX smoke/regression tests; no microphone, PulseAudio or user models.
Run: python3 -m analysis.testCryptoBridge
Requires existing sync/v2_timepack.py on the user's project.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave
import numpy as np
from crypto.frame_port import FrameKeySource


def _call(*args):
    env=dict(os.environ)
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    env['MKL_NUM_THREADS']='1'
    subprocess.run([sys.executable,'-m','experiment.cryptoTimepack20k',*map(str,args)],env=env,check=True,stdout=subprocess.DEVNULL)


def _write(path,arr,rate):
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(arr.astype('<i2').tobytes())


def _read(path):
    with wave.open(str(path),'rb') as w:
        return np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').copy()


def main():
    keys=FrameKeySource.for_demo_only(7342,'test')
    assert np.array_equal(keys.bits(2,160),keys.bits(2,160))
    assert np.array_equal(keys.bits(2,160),FrameKeySource.for_demo_only(7342,'test').bits(2,160))
    assert not np.array_equal(keys.bits(2,160),keys.bits(3,160))
    print('PASS key by absolute frame index; out-of-order recovery')
    with tempfile.TemporaryDirectory(prefix='nears_crypto_smoke_') as root:
        d=Path(root)
        # 320-sample * 240 frames = 4.8sec; no FFT reference limitations.
        s=np.arange(240*320)
        orig=np.rint(32767*.15*np.sin(2*np.pi*400*s/16000)).astype(np.int16)
        inp=d/'input.wav'
        _write(inp,orig,16000)
        for kind,extra in [('identity',[]),('plugin',['--plugin','crypto.example_sign_plugin:SignFlipDemo','--demo-seed','7342','--session','test'])]:
            tx=d/('tx_'+kind)
            rx=d/('rx_'+kind)
            _call('--role','tx','--algo',kind,'--input',inp,'--generate-only','--out',tx,*extra)
            _call('--role','rx','--algo',kind,'--wav',tx/'tx_encrypted_20k.wav','--out',rx,*extra)
            summ=json.loads((rx/'summary.json').read_text())
            recovered=_read(rx/'rx_decrypted_16k.wav')
            assert len(recovered)==len(orig)
            assert summ['frames_ok']==240 and summ['frames_lost']==0
            assert np.max(np.abs(recovered.astype('int32')-orig.astype('int32'))) <= 1
            print('PASS '+kind+' 240/240 frame integration, PCM restored')
        tx=d/'tx_identity'/'tx_encrypted_20k.wav'
        wire=_read(tx).reshape(-1,400)
        for action, count in [('drop',8),('drop',16),('insert',8),('insert',16)]:
            if action=='drop':
                damaged=np.concatenate((wire[:90],wire[90+count:]),axis=0)
            else:
                damaged=np.concatenate((wire[:90],np.zeros((count,400),dtype=np.int16),wire[90:]),axis=0)
            path=d/(action+str(count)+'.wav')
            _write(path,damaged.reshape(-1),20000)
            rx=d/(action+str(count)+'_rx')
            _call('--role','rx','--algo','identity','--wav',path,'--out',rx)
            summ=json.loads((rx/'summary.json').read_text())
            assert summ['last_frame']==239, summ
            if action=='drop':
                assert summ['frames_lost']==count, summ
            print('PASS',action,count,'last_frame=239','lost=',summ['frames_lost'])

        # Keys after a missing 16-frame segment must stay aligned to the absolute frame.
        wire_plugin=_read(d/'tx_plugin'/'tx_encrypted_20k.wav').reshape(-1,400)
        missing=np.concatenate((wire_plugin[:90],wire_plugin[106:]),axis=0)
        missing_file=d/'drop16_plugin.wav'
        _write(missing_file,missing.reshape(-1),20000)
        recover=d/'drop16_plugin_rx'
        _call('--role','rx','--algo','plugin','--plugin','crypto.example_sign_plugin:SignFlipDemo','--demo-seed','7342','--session','test','--wav',missing_file,'--out',recover)
        decrypted=_read(recover/'rx_decrypted_16k.wav')
        assert np.max(np.abs(decrypted[107*320:].astype('int32')-orig[107*320:].astype('int32')))<=1
        assert json.loads((recover/'summary.json').read_text())['frames_lost']==16
        print('PASS random-access HMAC key recovery after 16 lost frames')
    print('ALL SMOKE TESTS PASSED')

if __name__=='__main__':
    main()
