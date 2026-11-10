#!/usr/bin/env python3
"""Read-only ALG7 TX/RX diagnostics. Python 3.8+, stdlib + numpy.
Usage: python3 ALG7_TRIAGE.py --rx /tmp/alg7_call_rx [--tx /tmp/alg7_call_tx]
"""
import argparse
import csv
import json
import wave
from pathlib import Path
import numpy as np


def load_pcm16(path):
    if not path.is_file():
        return None, None
    with wave.open(str(path), 'rb') as f:
        channels, sampwidth, hz = f.getnchannels(), f.getsampwidth(), f.getframerate()
        if channels != 1 or sampwidth != 2:
            raise ValueError('Expected mono PCM16: ' + str(path))
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float64) / 32768., hz


def percentile(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q)) if len(x) else None


def stats(x):
    return 'rms=%.5f peak=%.4f clip%%=%.3f zero%%=%.2f' % (
        np.sqrt(np.mean(x*x)), np.max(np.abs(x)), 100*np.mean(np.abs(x)>.999), 100*np.mean(x==0))


def read_json(path):
    return json.loads(path.read_text('utf-8')) if path.is_file() else {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rx', type=Path, required=True)
    p.add_argument('--tx', type=Path)
    a=p.parse_args()
    root=a.rx
    if not root.is_dir():p.error('RX directory missing: '+str(root))
    summary=read_json(root/'summary.json')
    print('=== RX SUMMARY ===')
    for k in ('ok','lost','decoded','decode_errors','decode_cpu_sec','rx_samples', 'core_adjusted',
              'core_adjust_max_abs','estimated_guards','pn_gain_correction','pn_gain_final','last_frame'):
        if k in summary:print('%s: %s'%(k,summary[k]))
    rows=[]
    frames_path=root/'frames.csv'
    if frames_path.is_file():
        with frames_path.open(newline='', encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    print('\n=== RX FRAMES ===')
    if not rows:
        print('No frames.csv found');return
    idx=[];correct=[];valid=[];correction=[];estimated=0;corr=[];decode_ms=[];missing=0
    for r in rows:
        try:i=int(r.get('frame_index',r.get('frame','')))
        except ValueError:idx.append(None);continue
        idx.append(i)
        ok=r.get('status')=='ok';valid.append(ok)
        if ok and r.get('mode','correct')=='correct':correct.append(i)
        if r.get('guard_source')=='estimated':estimated+=1
        try:
            v=r.get('core_correction_samples','')
            if v not in ('',None):correction.append(int(float(v)))
        except ValueError:pass
        try:
            v=r.get('corr','')
            if v not in ('',None):corr.append(float(v))
        except ValueError:pass
        try:decode_ms.append(float(r.get('decode_ms','')))
        except ValueError:pass
    nonnull=[i for i in idx if i is not None]
    print('row count:',len(rows),'valid frames:',sum(valid),'invalid:',len(valid)-sum(valid))
    print('index range:',min(nonnull) if nonnull else None,'to',max(nonnull) if nonnull else None)
    if nonnull:print('first indices:',nonnull[:16]);print('last indices:',nonnull[-16:])
    badseq=sum(b<=a for a,b in zip(nonnull,nonnull[1:]))
    if badseq:print('WARNING: non-increasing indices:',badseq)
    if corr:print('PN corr p10/median/p90:',*[round(percentile(corr,q),4) for q in (10,50,90)])
    if correction:print('core timing corrections nonzero:',sum(x!=0 for x in correction),'of',len(correction),'max:',max(abs(x) for x in correction))
    print('estimated guard rows:',estimated)
    if decode_ms:print('decoder ms p50/p90/p99:',*[round(percentile(decode_ms,q),3) for q in (50,90,99)])
    if summary.get('decoded'):
        cpu=summary.get('decode_cpu_sec',0)
        n=summary['decoded'];print('average decode ms:',round(1000*cpu/n,3),'frame budget: 20ms')
        print('decoder CPU / frame budget:',round(cpu/(n*0.02),3))
    print('frames in CORRECT and status OK:',len(correct))
    c,sr=load_pcm16(root/'rx_cipher_16k.wav');d,dr=load_pcm16(root/'rx_listened_16k.wav')
    for name,x,fs in [('rx_cipher_16k',c,sr),('rx_listened_16k',d,dr)]:
        if x is not None and len(x):print(name,fs,'Hz',len(x)//320,'frames',stats(x))
    incoming,rate=load_pcm16(root/'rx_incoming_20k.wav')
    if incoming is not None and len(incoming):print('rx_incoming_20k',rate,'Hz', stats(incoming))
    if a.tx is None:return
    print('\n=== TX/ RX PAYLOAD COMPARISON ===')
    tx=read_json(a.tx/'summary.json')
    for k in ('frames','encode_cpu_sec','input_samples_16k','padded_samples_16k'):
        if k in tx:print('tx %s: %s'%(k,tx[k]))
    if tx.get('frames'):
        print('TX encode CPU/frame:',round(1000*tx.get('encode_cpu_sec',0)/tx['frames'],2),'ms; budget 20ms per frame')
    t,fr=load_pcm16(a.tx/'tx_wire_20k.wav')
    if t is None or c is None:
        print('Missing tx_wire_20k.wav or rx_cipher_16k.wav');return
    if len(t)%400 or len(c)%320:print('Warning: partial frame in WAV')
    t=t[:len(t)//400*400].reshape(-1,400)[:,:320]
    r=c[:len(c)//320*320].reshape(-1,320)
    good=[]
    for j,row in enumerate(rows):
        if j>=len(r):break
        i=idx[j] if j<len(idx) else None
        if row.get('status')!='ok' or i is None or i<0 or i>=len(t):continue
        x=t[i];y=r[j]
        if np.std(x)<1e-6 or np.std(y)<1e-6:continue
        good.append((x,y))
    if not good:
        print('No index-matched non-silent frames. Check RX initial frame epoch, routing, timestamps.')
        return
    A=np.concatenate([item[0] for item in good]); B=np.concatenate([item[1] for item in good]);
    r0=float(np.corrcoef(A,B)[0,1]);k=float(np.dot(A,B)/max(1e-20,np.dot(A,A)))
    err=np.sqrt(np.mean((B-k*A)**2));den=np.sqrt(np.mean(B*B));
    print('matched frames:',len(good),'out of',len(rows))
    print('cipher correlation TX↔RX:',round(r0,5),'(1 is ideal)')
    print('best constant gain RX/TX:',round(k,5),'(affected by payload-gain setting)')
    print('relative error after fitting gain:',round(float(err/max(den,1e-20)),5))
    if r0<0.7:print('WARNING: ciphertext strongly modified or index alignment invalid')
    elif r0<0.98:print('WARNING: noticeable ciphertext distortion')
    else:print('Ciphertext waveform appears well preserved; inspect ALG7 parameter recovery and synthesis.')

if __name__=='__main__':main()
