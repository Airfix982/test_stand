#!/usr/bin/env python3
"""Compare ALG6 WHT encrypted PCM BEFORE/AFTER VDO.Ninja by frameIndex.
No edits to encryption, adapter, SuperSpinner or runtime. Run from project root:
python3 -m algoss.alg6_wht.diagnose_channel --tx /tmp/alg6_tx --rx /tmp/alg6_corr_rx
"""
import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np


def audio(path, hz):
    with wave.open(str(path), 'rb') as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if fmt != (1, 2, hz):
            raise ValueError(f'{path}: expected PCM16 mono {hz} Hz, got {fmt}')
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').copy()


def cosine(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den > 0 else 0.


def cosine_pairs(tx, rx, rows, shift, rx_offset=0, first=0, last=None):
    """Median correlation at a candidate TX epoch and RX frame offset."""
    last = len(rows) if last is None else last
    values = []
    for j in range(first, last):
        if rows[j]['status'] != 'ok':
            continue
        i, k = int(rows[j]['frame']) + shift, j + rx_offset
        if 0 <= i < len(tx) and 0 <= k < len(rx):
            values.append(cosine(tx[i, :256], rx[k, :256]))
    return float(np.median(values)) if values else -1.0


def find_shift(tx, rx, rows, limit):
    # Test both possible positions of an extra RX WAV frame.
    # Evaluate prefix and suffix separately: a missing CSV row can be in the middle.
    starts = [(0, min(32, len(rows)))]
    if len(rows) > 32:
        starts.append((max(0, len(rows) - 32), len(rows)))
    offsets = [0, 1] if len(rx) == len(rows) + 1 else [0]
    scores = []
    for shift in range(-min(limit, 128), min(limit, len(tx) - 1) + 1):
        candidates = [cosine_pairs(tx, rx, rows, shift, off, lo, hi)
                      for lo, hi in starts for off in offsets]
        score = max(candidates)
        if score > -1.0:
            scores.append((score,shift))
    if not scores:
        raise ValueError('No valid frames to estimate TX/RX frame offset')
    maximum = max(s for s, _ in scores)
    # A one-frame discrepancy can be explained equally well as an extra RX
    # prefix + shift 0 OR as an extra RX suffix + TX shift -1.
    # The SuperSpinner epoch normally matches TX, so prefer shift 0 when tied.
    candidates = [(v,k) for v,k in scores if v >= maximum - 0.02]
    v,k = min(candidates,key=lambda item:(abs(item[1]),-item[0]))
    return v,k


def align_rx(tx, rx, rows, shift):
    """Return CSV-aligned RX WAV frames and alignment explanation.

    With 1 extra WAV frame, score every possible insertion position using
    encrypted PCM correlation with TX. This avoids silently shifting every
    subsequent row after an unlogged RX callback.
    """
    nr, nw = len(rows), len(rx)
    if nr == nw:
        return rx, {'rx_wav_extra_frames': 0, 'rx_skipped_wav_index': None,
                    'alignment_method': 'one-to-one', 'alignment_median_correlation':
                    round(cosine_pairs(tx, rx, rows, shift), 6)}
    if nw != nr + 1:
        raise ValueError(f'Cannot safely align frames.csv ({nr}) and rx_cipher_16k.wav ({nw}): '
                         'only one extra RX WAV frame is supported. Preserve files and inspect CSV.')

    # a[j] matches rx[j], b[j] matches rx[j+1]. Choose k maximizing
    # sum(a[:k]) + sum(b[k:]): RX[k] is the extra unmatched audio frame.
    a = np.zeros(nr, dtype=np.float64)
    b = np.zeros(nr, dtype=np.float64)
    valid = np.zeros(nr, dtype=bool)
    for j, row in enumerate(rows):
        if row['status'] != 'ok':
            continue
        f = int(row['frame']) + shift
        if not 0 <= f < len(tx):
            continue
        valid[j] = True
        a[j] = cosine(tx[f, :256], rx[j, :256])
        b[j] = cosine(tx[f, :256], rx[j + 1, :256])
    scores = np.r_[0.0, np.cumsum(a)] + np.r_[np.cumsum(b[::-1])[::-1], 0.0]
    drop = int(np.argmax(scores))
    # If correlation is too weak to locate one unmatched frame, prefer the
    # physically likely trailing WAV frame, and mark this as uncertain.
    baseline = float(scores[-1]); gain = float(scores[drop] - baseline)
    count = int(valid.sum())
    confidence = 'correlation-based' if count and gain > max(.1, .02 * count) else 'uncertain, assuming last RX frame'
    if confidence.startswith('uncertain'):
        drop = nr
    aligned = np.concatenate((rx[:drop], rx[drop+1:]), axis=0)
    c = [cosine(tx[int(rows[j]['frame'])+shift, :256], aligned[j, :256])
         for j in range(nr) if valid[j]]
    return aligned, {
        'rx_wav_extra_frames': 1,
        'rx_skipped_wav_index': drop,
        'alignment_method': confidence,
        'alignment_gain_vs_dropping_last': round(gain, 5),
        'alignment_median_correlation': round(float(np.median(c)), 6) if c else None,
    }


def collect(tx, rx, rows, shift):
    pairs = []
    for row, data in zip(rows, rx):
        if row['status'] != 'ok':
            continue
        ix = int(row['frame']) + shift
        if 0 <= ix < len(tx):
            pairs.append((int(row['frame']), tx[ix, :320], data))
    return pairs


def main():
    p = argparse.ArgumentParser(description='ALG6 VDO.Ninja WHT ciphertext corruption diagnosis')
    p.add_argument('--tx', type=Path, required=True, help='TX folder with tx_cipher_20k.wav')
    p.add_argument('--rx', type=Path, required=True, help='RX folder with rx_cipher_16k.wav and frames.csv')
    p.add_argument('--shift', type=int, help='force TX frame index = RX frameIndex + shift')
    p.add_argument('--max-shift', type=int, default=256)
    p.add_argument('--output', type=Path, help='diagnostic csv; default RX/wht_channel_comparison.csv')
    a = p.parse_args()
    tx_path = a.tx / 'tx_cipher_20k.wav'
    rx_path = a.rx / 'rx_cipher_16k.wav'
    csv_path = a.rx / 'frames.csv'
    tx = audio(tx_path, 20000)
    rx = audio(rx_path, 16000)
    if len(tx) % 400 or len(rx) % 320:
        raise ValueError(f'Unexpected lengths: TX {len(tx)} samples (need /400), RX {len(rx)} (need /320)')
    tx, rx = tx.reshape(-1, 400), rx.reshape(-1, 320)
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if abs(len(rows) - len(rx)):
        print(f'NOTICE: RX CSV rows={len(rows)}, RX WAV frames={len(rx)}. '
              'Trying alignment without changing WAV/CSV files.')
    if a.shift is None:
        score, guessed = find_shift(tx, rx, rows, a.max_shift)
        shift = guessed
    else:
        guessed, shift = a.shift, a.shift
        score = cosine_pairs(tx, rx, rows, shift)
    rx_aligned, alignment = align_rx(tx, rx, rows, shift)
    print('ALIGNMENT:', json.dumps(alignment, ensure_ascii=False))
    pairs = collect(tx, rx_aligned, rows, shift)
    if not pairs:
        raise ValueError('No comparable TX/RX frames. Check --shift, --tx and --rx')

    out = a.output or (a.rx / 'wht_channel_comparison.csv')
    bit_errors = np.zeros((), dtype=np.int64)
    word_errors = np.zeros((), dtype=np.int64)
    sample_errors = np.zeros((), dtype=np.int64)
    absolute = []
    tail_rms = []
    sample_corr = []
    with out.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rx_frame', 'tx_frame', 'equal_data_samples/256', 'equal_words/128',
                    'bit_errors/4096', 'mean_abs_pcm_delta', 'data_correlation', 'tail_rms_pcm'])
        for frame, orig, received in pairs:
            # 256 samples contain 128 x 32-bit WHT encrypted words, last 64 = zero padding.
            x = orig[:256]
            y = received[:256]
            changed = (x != y)
            word_differences = (x.view('<i4') != y.view('<i4')).sum()
            bitcount = int(np.unpackbits(np.bitwise_xor(x, y).view(np.uint8)).sum())
            abs_diff = np.abs(x.astype(np.int32) - y.astype(np.int32))
            correlation = cosine(x, y)
            tail = float(np.sqrt(np.mean(received[256:].astype(np.float64)**2)))
            bit_errors += bitcount
            word_errors += int(word_differences)
            sample_errors += int(changed.sum())
            absolute.append(float(abs_diff.mean()))
            tail_rms.append(tail)
            sample_corr.append(correlation)
            w.writerow([frame, frame + shift, 256 - int(changed.sum()),
                        128 - int(word_differences), bitcount,
                        round(float(abs_diff.mean()), 3), round(correlation, 5), round(tail, 3)])
    n = len(pairs)
    result = {
        'tx_frames': len(tx), 'rx_frames': len(rx), 'rx_csv_rows': len(rows),
        'compared_ok_frames': n, **alignment,
        'best_shift_guess': guessed, 'best_shift_score': round(score, 6),
        'applied_shift': shift, 'median_data_correlation': round(float(np.median(sample_corr)), 6),
        'exact_cipher_samples_pct': round(100 * (1 - int(sample_errors)/(256 * n)), 5),
        'exact_cipher_words_pct': round(100 * (1 - int(word_errors)/(128 * n)), 5),
        'cipher_bit_error_pct': round(100 * int(bit_errors)/(4096 * n), 5),
        'average_abs_pcm_delta': round(float(np.mean(absolute)), 4),
        'average_padding_tail_rms_pcm': round(float(np.mean(tail_rms)), 4),
        'details_csv': str(out),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    (a.rx / 'wht_channel_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    if score < 0.25:
        print('WARNING: frame offset is unreliable (channel badly distorted or different TX/RX runs).')
    if alignment['alignment_method'].startswith('uncertain'):
        print('WARNING: extra RX frame position not proven. Alignment may shift after missing CSV row.')
    if result['exact_cipher_words_pct'] < 100:
        print('FAIL: encrypted WHT uint32 words changed before decrypt_chunk().')
    else:
        print('PASS: all compared WHT ciphertext words arrived bit-exact.')


if __name__ == '__main__':
    main()
