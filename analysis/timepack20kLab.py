"""Time-packed SuperSpinner experiment; NOT bit-perfect transport through a speech codec.

TX reads 320 audio samples at 16k (20ms), plays them unaltered at 20k (16ms),
then adds 80 PN samples at 20k (4ms). RX captures 20k, detects PN markers,
extracts the first 320 samples and writes them at 16k (20ms).

Do not use resample_poly(320, 400) on the payload: ordinary SRC would occupy
all 400 samples and leave no room for a guard.
"""
import argparse
from pathlib import Path
import wave

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly
from sync.v2_timepack import makeGuardBank

SOURCE_RATE = 16000
WIRE_RATE = 20000
PAYLOAD = 320
GUARD = 80
FRAME = 400
CYCLE = 64


def load_wav(path, rate):
    with wave.open(str(path), 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (rate, 1, 2):
            raise ValueError('Expected mono 16-bit WAV at %d Hz: %s' % (rate, path))
        return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0


def save_wav(path, x, rate):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x16 = np.rint(np.clip(x, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(x16.tobytes())


def generate(args):
    if args.input:
        payload = load_wav(args.input, SOURCE_RATE)
        if args.duration:
            payload = payload[:int(round(args.duration * SOURCE_RATE))]
        nf = (len(payload) + PAYLOAD - 1) // PAYLOAD
        payload = np.pad(payload, (0, nf * PAYLOAD - len(payload)))
    else:
        duration = 20.0 if args.duration is None else args.duration
        nf = int(round(duration * 50))
        rng = np.random.default_rng(args.seed)
        taps = firwin(63, [250, 3300], pass_zero=False, fs=SOURCE_RATE)
        white = rng.standard_normal(nf * PAYLOAD + 128)
        payload = lfilter(taps, [1.0], white)[128:].astype(np.float32)
        payload *= args.level / (np.sqrt(np.mean(payload * payload)) + 1e-12)
    if nf < 1:
        raise ValueError('Need >= 1 frame')
    original = np.clip(payload.reshape(nf, PAYLOAD), -0.90, 0.90)
    guards = makeGuardBank(WIRE_RATE, GUARD, CYCLE)
    wire = np.empty((nf, FRAME), dtype=np.float32)
    wire[:, :PAYLOAD] = original
    wire[:, PAYLOAD:] = args.pn_amp * guards[np.arange(nf) % CYCLE]
    if float(np.max(np.abs(wire))) > 1.0:
        raise ValueError('PN or payload clips; reduce --pn-amp or --level')
    save_wav(args.out, wire.ravel(), WIRE_RATE)
    save_wav(args.reference, original.ravel(), SOURCE_RATE)
    print('TX: %d frames, %.3fs, source=16k, wire=20k, format=[320 payload][80 PN]' % (nf, nf/50))
    print('input payload RMS %.5f, guard amplitude %.4f, TX peak %.4f' % (float(np.sqrt(np.mean(original ** 2))), args.pn_amp, float(np.max(np.abs(wire)))))
    print('TX file: %s   reference file: %s' % (args.out, args.reference))


def simulate(args):
    wire = load_wav(args.input, WIRE_RATE)
    # Simulate merely the sample-rate conversions of the 48k pipeline, NOT Opus.
    via48 = resample_poly(wire, 12, 5)
    back20 = resample_poly(via48, 5, 12)
    save_wav(args.out, back20, WIRE_RATE)
    print('Simulated SRC: 20k -> 48k -> 20k; not a Telemost/Opus simulation')


def verify(args):
    reference = load_wav(args.reference, SOURCE_RATE)
    recovered = load_wav(args.rx, SOURCE_RATE)
    n = min(len(reference), len(recovered)) // PAYLOAD
    if n == 0: raise ValueError('Recovered file has no aligned 20ms frames')
    a = reference[:n*PAYLOAD].reshape(n, PAYLOAD).astype(np.float64)
    b = recovered[:n*PAYLOAD].reshape(n, PAYLOAD).astype(np.float64)
    active = np.sum(b*b, axis=1) > 1e-9
    if args.frames_csv:
        import csv
        stat = {}
        with open(args.frames_csv, newline='') as f:
            for row in csv.DictReader(f):
                stat[int(row['frame'])] = row['status']
        active &= np.array([stat.get(i) == 'ok' for i in range(n)])
    if not np.any(active):
        print('No matching OK/nonzero frames. Check RX summary/events and routing.')
        return
    x, y = a[active].ravel(), b[active].ravel()
    exact = int(np.sum(np.all(a[active] == b[active], axis=1)))
    snr = 10*np.log10(np.mean(x*x) / (np.mean((x-y)**2) + 1e-20))
    gain = float(np.dot(x,y) / (np.dot(y,y)+1e-20))
    adj = 10*np.log10(np.mean(x*x)/(np.mean((x-gain*y)**2)+1e-20))
    corr = float(np.corrcoef(x,y)[0,1])
    print('REFERENCE frames:', len(reference)//PAYLOAD, 'RX frames:',len(recovered)//PAYLOAD)
    print('OK/nonzero frames analyzed:', int(active.sum()), '/', n)
    print('PAYLOAD SNR: %.2f dB (raw), %.2f dB (optimal gain %.3f)' % (snr, adj, gain))
    print('PAYLOAD Pearson correlation: %.4f; byte-exact frames: %d / %d' % (corr,exact,int(active.sum())))
    print('Note: bit-exact ciphertext transmission requires a modem/FEC; SNR alone is insufficient.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--duration', type=float, default=None)
    g.add_argument('--seed', type=int, default=20261019)
    g.add_argument('--level', type=float, default=0.06)
    g.add_argument('--pn-amp', type=float, default=0.18)
    g.add_argument('--input', default=None, help='Optional real source WAV at 16000Hz/mono/s16')
    g.add_argument('--out', default='/tmp/tp20k_tx.wav')
    g.add_argument('--reference', default='/tmp/tp20k_ref.wav')
    s = sub.add_parser('simulate')
    s.add_argument('--input', required=True)
    s.add_argument('--out', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--reference', required=True)
    v.add_argument('--rx', required=True)
    v.add_argument('--frames-csv', default=None)
    args = p.parse_args()
    {'generate': generate, 'simulate': simulate, 'verify': verify}[args.mode](args)

if __name__ == '__main__': main()
