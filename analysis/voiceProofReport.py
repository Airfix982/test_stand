"""Compare received cipher/correct/wrong with the actual TX microphone PCM.

Absolute-frame verdict is based on same-index (zero-lag) correlation;
best-lag measurement is reported separately and cannot imply correct PN epoch.
"""
import argparse
import csv
import json
from pathlib import Path
import wave
import numpy as np
from scipy.signal import correlate, correlation_lags

FILES={'cipher':'rx_cipher_payload_16k.wav','correct':'rx_decrypted_16k.wav','wrong':'rx_wrong_index_16k.wav'}


def read_mono(path):
    with wave.open(str(path),'rb') as f:
        if f.getframerate()!=16000 or f.getnchannels()!=1 or f.getsampwidth()!=2:
            raise ValueError('%s is not 16k mono s16 WAV' % path)
        return np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float64)/32768.0


def metric(ref,got,offset=0,active=None):
    # Positive offset: got begins offset samples after ref begins.
    rstart=max(0,-offset);gstart=max(0,offset)
    n=min(len(ref)-rstart,len(got)-gstart)
    if n<320:return {'pearson':None,'snr_db':None,'samples':n}
    r=ref[rstart:rstart+n];g=got[gstart:gstart+n]
    if active is not None:
        mask=active[rstart:rstart+n]
        r=r[mask];g=g[mask]
    if len(r)<320 or np.std(r)<1e-5 or np.std(g)<1e-7:
        return {'pearson':None,'snr_db':None,'samples':len(r)}
    corr=float(np.corrcoef(r,g)[0,1])
    gain=float(np.dot(r,g)/max(np.dot(g,g),1e-15))
    residual=r-g*gain
    snr=float(10*np.log10(max(np.mean(r*r),1e-15)/max(np.mean(residual*residual),1e-15)))
    return {'pearson':round(corr,5),'snr_opt_gain_db':round(snr,3),'samples':len(r),'gain':round(gain,4)}


def best_lag(ref,got,search_samples=16000*2):
    # Bias toward speech-active sections, but never use the lag to claim frame correctness.
    n=min(len(ref),len(got))
    r=ref[:n]-np.mean(ref[:n]);g=got[:n]-np.mean(got[:n])
    if n<320:return None
    c=correlate(g,r,mode='full',method='fft')
    l=correlation_lags(len(g),len(r),mode='full')
    lo=np.searchsorted(l,-search_samples);hi=np.searchsorted(l,search_samples,side='right')
    if lo>=hi:return None
    return int(l[lo+int(np.argmax(c[lo:hi]))])


def main():
    p=argparse.ArgumentParser(description='Absolute-frame and A/B/C audio diagnostic')
    p.add_argument('--rx-dir',required=True)
    p.add_argument('--tx',help='Actual TX tx_plain_16k.wav. Without it, exact recovery CANNOT be verified')
    p.add_argument('--out',help='JSON report path (default RX/voiceproof_report.json)')
    args=p.parse_args();root=Path(args.rx_dir)
    branch={k:read_mono(root/f) for k,f in FILES.items()}
    frames=[]
    with open(root/'frames.csv',newline='') as f:
        frames=list(csv.DictReader(f))
    indices=[int(row['frame']) for row in frames]
    ok=[int(row['frame']) for row in frames if row['status']=='ok']
    index_gaps=[(a,b) for a,b in zip(indices,indices[1:]) if b!=a+1]
    info={'frames_output':len(frames),'frames_ok':len(ok),'frames_lost':len(frames)-len(ok),
          'first_index':indices[0] if indices else None,'last_index':indices[-1] if indices else None,
          'nonconsecutive_indices':index_gaps[:10]}
    summary_path=root/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        info['wrong_offset_in_transport_frames']=summary.get('wrong_decoding_absolute_offset')
        info['sync_frames_lost']=summary.get('frames_lost')
        info['sync_reacquisitions']=summary.get('reacquisitions')
    if not args.tx:
        info['verdict']='NO_GROUND_TRUTH - supply TX/tx_plain_16k.wav to establish correct vs wrong'
        print(json.dumps(info,indent=2))
    else:
        ref=read_mono(args.tx)
        n=len(ref)
        act=np.zeros(n,dtype=bool)
        for pos in range(0,n,320):
            part=ref[pos:pos+320]
            if len(part) and np.sqrt(np.mean(part*part))>0.003:
                act[pos:pos+len(part)]=True
        info['tx_frames']=int(np.ceil(len(ref)/320))
        info['active_tx_frames']=int(np.count_nonzero(act)//320)
        info['zero_lag']={k:metric(ref,v,0,act) for k,v in branch.items()}
        lag=best_lag(ref,branch['correct'],search_samples=32000)
        info['correct_best_lag_samples']=lag
        info['best_lag']=({k:metric(ref,v,lag,act) for k,v in branch.items()} if lag is not None else None)
        raw=info['zero_lag']
        a=raw['correct']['pearson'];b=raw['cipher']['pearson'];c=raw['wrong']['pearson']
        if a is None or b is None or c is None:
            info['verdict']='INCONCLUSIVE (insufficient signal / incomplete recording)'
        elif a>=0.55 and a-max(abs(b),abs(c))>=0.30 and not index_gaps:
            info['verdict']='PASS: correct frame index, correct > cipher and wrong on same TX timeline'
        elif info['best_lag'] and info['best_lag']['correct']['pearson'] is not None and info['best_lag']['correct']['pearson']>=0.55:
            info['verdict']='AUDIO_RECOVERED_BUT_INDEX_NOT_PROVEN (best lag is not zero)'
        else:
            info['verdict']='FAIL_OR_INCONCLUSIVE: correct recovery not sufficiently distinguished'
        print(json.dumps(info,indent=2,ensure_ascii=False))
    path=Path(args.out) if args.out else root/'voiceproof_report.json'
    path.write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    print('Report:',path)

if __name__=='__main__':main()