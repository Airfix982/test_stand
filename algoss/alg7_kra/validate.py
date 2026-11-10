"""ALG7 prerequisites + deterministic key schedule + SuperSpinner selftests.

Does not modify model files, does not invoke speech_encoder unless --full.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import numpy as np


def prerequisites(directory):
    files = (
        'speech_model.npz', 'pseudospeech_model.npz', 'spherical_code.npz',
        'speech_encoder', 'cpp/lib/single_pitch.so', 'speech_analyzer.py',
        'pseudo_speech.py', 'encryption.py', 'single_pitch.py',
        'DirectLPCSynthesizer.py'
    )
    missing = []
    for name in files:
        path = directory / name
        good = path.is_file()
        print(('[OK] ' if good else '[MISSING] ') + str(path))
        if not good:missing.append(name)
    enc=directory/'speech_encoder'
    if enc.exists() and not os.access(str(enc),os.X_OK):
        print('[ERROR] speech_encoder is not executable: chmod +x '+str(enc))
        missing.append('speech_encoder (chmod +x)')
    for prog in ('pactl','parec','paplay','ffmpeg'):
        if shutil.which(prog):print('[OK] command',prog)
        else:print('[MISSING] command',prog);missing.append('command '+prog)
    if not missing:
        try:
            with np.load(str(directory/'speech_model.npz')) as model:
                frame=int(model['speech_frame_size'])
            with np.load(str(directory/'pseudospeech_model.npz')) as model:
                fs=int(model['Fs']);flng=int(model['flng']);glng=int(model['glng'])
            with np.load(str(directory/'spherical_code.npz')) as model:
                bits=model['encryption_bits']
                c=np.asarray(model['c_0']).reshape(-1)
                u=np.asarray(model['u_0']).reshape(-1)
            print('MODEL INFO: analyzer frame=%d @ 8k; pseudo Fs=%d frame=%d+%d; bits=%s; c0=%s u0=%s' %
                  (frame,fs,flng,glng,bits, c.shape,u.shape))
            if (frame,fs,flng,glng)!=(160,16000,240,80):
                missing.append('model frame geometry mismatch (need 160/16000/240/80)')
        except Exception as exc:
            print('[ERROR] model inspect:',repr(exc));missing.append('model file error')
    return missing


def check_rng(seed=37,bits=174):
    from crypto.alg7_kra_adaptor import OriginalNumpyRngCursor
    cursor=OriginalNumpyRngCursor(seed,bits)
    orig=np.random.default_rng(seed)
    keys=[orig.integers(0,2,size=bits,dtype=np.uint8) for _ in range(150)]
    for i in (0,1,2,37,37,63,65,149,80,2,149):
        got=cursor.get_key_for_frame(i)
        if not np.array_equal(got,keys[i]):raise AssertionError('RNG mismatch at frame '+str(i))
    print('[PASS] original default_rng(seed=%d) cursor with sequential, skip, repeated and rewind frames' % seed)


def check_sync():
    from sync.v2_timepack import FixedGuardSyncV1,SyncV1Config,makeGuardBank
    cfg=SyncV1Config(rate=20000,frameSamples=400,coreSamples=320,
                     guardSamples=80,spinnerSize=8,superSpinnerSize=8)
    guards=makeGuardBank(20000,80,64)
    rng=np.random.default_rng(512)
    voice=rng.standard_normal((64,320)).astype(np.float32)*.06
    wire=np.concatenate((voice,guards*.12),axis=1).astype(np.float32).reshape(-1)
    got=[]
    sync=FixedGuardSyncV1(cfg,lambda index,packet,status,meta:got.append((index,status,packet[:320].copy())),lambda ev:None)
    for j in range(0,len(wire),173):sync.push(wire[j:j+173])
    sync.finish()
    if len(got)!=64 or any(i!=j or s!='ok' or np.max(np.abs(y-voice[j]))>1e-6 for j,(i,s,y) in enumerate(got)):
        raise AssertionError('SuperSpinner 320+80 fails on unchanged PCM')
    print('[PASS] SuperSpinner v2: 64/64 correct index, payload sample boundaries on reference PCM')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--algo-dir',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--seed',type=int,default=37)
    p.add_argument('--no-model-check',action='store_true')
    a=p.parse_args()
    bits = 174  # only a fallback when the genuine model is unavailable
    code = a.algo_dir.resolve()/'spherical_code.npz'
    if code.is_file():
        try:
            with np.load(str(code)) as model:
                bits=int(model['encryption_bits'][2])
        except Exception as e:
            print('[WARN] cannot read encryption_bits from spherical code:',repr(e))
    check_rng(a.seed,bits);check_sync()
    if not a.no_model_check:
        absent=prerequisites(a.algo_dir.resolve())
        if absent:
            print('[BLOCKED] Missing original ALG7 runtime assets:',', '.join(absent))
            raise SystemExit(2)
    print('[PASS] ALG7 static prerequisites; this is NOT an end-to-end model/audio validation')

if __name__=='__main__':main()
