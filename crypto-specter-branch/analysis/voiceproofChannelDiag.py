#!/usr/bin/env python3
"""Diagnose TimePack+VoiceProof from a single live transmission, no cipher changes.

Requires TX --record-wire and RX WAV/frames.csv from same session.
The absolute-frame comparisons use frames.csv indices, not a guessed alignment.
Correlations measure PCM survival and do not by themselves certify a secure cipher.
"""
import argparse
import csv
import json
import math
import wave
from pathlib import Path

import numpy as np


def readwav(path, rate):
    with wave.open(str(path), 'rb') as f:
        if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (rate, 1, 2):
            raise ValueError('%s: expected %d Hz, mono s16le' % (path, rate))
        return np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float64) / 32768


def corr(a, b):
    if len(a) < 160:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    aa = np.linalg.norm(a)
    bb = np.linalg.norm(b)
    return float(np.dot(a, b) / aa / bb) if aa > 1e-9 and bb > 1e-9 else None


def rms(a):
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def dbfs(a):
    return round(20 * math.log10(max(rms(a), 1e-10)), 2)


def metric_pairs(ref, got):
    if not ref:
        return {'frames': 0, 'pearson': None, 'median_frame_pearson': None, 'positive_matches_over_0_5': 0}
    ref = np.concatenate(ref)
    got = np.concatenate(got)
    n = len(ref) // 320
    a = ref.reshape(n, 320)
    b = got.reshape(n, 320)
    corrs = [corr(x, y) for x, y in zip(a, b)]
    good = [z for z in corrs if z is not None]
    return {'frames': n, 'pearson': round(corr(ref, got), 4) if corr(ref, got) is not None else None,
            'median_frame_pearson': round(float(np.median(good)), 4) if good else None,
            'positive_matches_over_0_5': int(sum(z > 0.5 for z in good)),
            'rx_to_tx_rms_ratio': round(rms(got) / max(rms(ref), 1e-9), 4)}


def main():
    p = argparse.ArgumentParser(description='Diagnose TimePack ciphertext, frame index and VoiceProof inverse using TX/RX recordings')
    p.add_argument('--tx-dir', required=True)
    p.add_argument('--rx-dir', required=True)
    p.add_argument('--gain', type=float, default=0.75, help='TX --tx-gain, default 0.75')
    args = p.parse_args()
    tx, rx = Path(args.tx_dir), Path(args.rx_dir)
    wire = readwav(tx/'tx_encrypted_20k.wav', 20000)
    plain = readwav(tx/'tx_plain_16k.wav', 16000)
    cipher = readwav(rx/'rx_cipher_payload_16k.wav', 16000)
    correct = readwav(rx/'rx_decrypted_16k.wav', 16000)
    wrong = readwav(rx/'rx_wrong_index_16k.wav', 16000)
    with (rx/'frames.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    nout = min(len(rows), len(cipher)//320, len(correct)//320, len(wrong)//320)
    if not nout:
        raise SystemExit('No frames in RX output')
    tx_frames = min(len(wire)//400, len(plain)//320)
    # Most active half of original speech, excluding silence; pick only independent ACTIVE frames.
    voice_rms = np.sqrt(np.mean(plain[:tx_frames*320].reshape(tx_frames, 320)**2, axis=1))
    cutoff = max(0.002, float(np.quantile(voice_rms, 0.50)))
    def collect(ref_wave, rx_wave, offset=0, is_wire=False):
        a,b=[],[]
        for j,row in enumerate(rows[:nout]):
            if row.get('status') != 'ok':
                continue
            try: logical = int(row['frame'])
            except (KeyError,ValueError): continue
            target = logical + offset
            if target < 0 or target >= tx_frames or voice_rms[target] < cutoff:
                continue
            r = ref_wave[target*400:target*400+320] / args.gain if is_wire else ref_wave[target*320:target*320+320]
            g = rx_wave[j*320:j*320+320]
            if len(r)==320 and len(g)==320:
                a.append(r)
                b.append(g)
        return metric_pairs(a,b)
    shifts = {str(d):collect(wire,cipher,d,True) for d in (-128,-64,0,64,128)}
    dec = {name:collect(plain,arr) for name,arr in [('cipher',cipher),('correct',correct),('wrong',wrong)]}
    summary = {}
    if (rx/'summary.json').exists():
        summary = json.loads((rx/'summary.json').read_text())
    z = shifts['0']['pearson']
    c = dec['correct']['pearson']
    w = dec['wrong']['pearson']
    other = [(int(k),v['pearson']) for k,v in shifts.items() if k!='0' and v['pearson'] is not None]
    best_other = max(other,key=lambda x:x[1]) if other else None
    if shifts['0']['frames'] < 50:
        verdict = 'INSUFFICIENT_ACTIVE_FRAMES: need >50 speech-active decoded frames'
    elif c is not None and c > 0.65 and (w is None or c > w+0.3):
        verdict = 'PASS: received ciphertext preserved; correct decryption distinctly recovers TX voice'
    elif z is not None and z > 0.35 and (best_other is None or z > best_other[1]+0.15):
        verdict = 'CIPHERTEXT_OK_AT_INDEX_0: frame index looks correct; check TX/RX key bytes, session and grain if decrypt is noise'
    elif best_other is not None and best_other[1] > 0.35 and (z is None or best_other[1] > z+0.15):
        verdict = 'POSSIBLE_EPOCH_SHIFT_%+d: ciphertext corresponds to different TX frame index' % best_other[0]
    else:
        verdict = 'CIPHERTEXT_NOT_PRESERVED_OR_OFFSET_UNRESOLVED: likely WebRTC processing/noise suppression; cannot conclude PN index failure'
    result = {'tx_frames':tx_frames,'rx_output_frames':nout,'active_threshold_rms':round(cutoff,6),
              'rx_frames_ok':summary.get('frames_ok'),'rx_frames_lost':summary.get('frames_lost'),
              'rx_reacquisitions':summary.get('reacquisitions'),
              'rms_dbfs':{'tx_plain':dbfs(plain),'tx_cipher_payload':dbfs(wire[:tx_frames*400].reshape(tx_frames,400)[:,:320].reshape(-1)/args.gain),
                          'rx_cipher':dbfs(cipher),'rx_correct':dbfs(correct),'rx_wrong':dbfs(wrong)},
              'cipher_vs_wire_by_absolute_index':shifts,
              'branch_vs_tx_plain_same_absolute_index':dec,
              'verdict':verdict,
              'limits':'Only compares active voice frame positions, with no hidden time alignment. PCM can be distorted while PN remains correct.'}
    path = rx/'voiceproof_channel_diag.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print('\nREPORT:',path)

if __name__=='__main__':
    main()