import argparse, csv, itertools, wave
from collections import Counter
from pathlib import Path
import numpy as np

RATE, FRAME, CORE, GUARD = 16000, 320, 240, 80
BINS = range(7, 53)


def codes(count, seed):
    choices = [(a, b, c) for a, b, c in itertools.combinations(BINS, 3) if b - a >= 5 and c - b >= 5]
    if count > len(choices):
        raise ValueError(f"Maximum duration = {len(choices) / 50:.1f}s")
    rng = np.random.default_rng(seed)
    return [choices[int(i)] for i in rng.permutation(len(choices))[:count]]


def load_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != RATE:
            raise ValueError("Expected WAV mono, 16 kHz, s16")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    return x.astype(np.float32) / 32768.0


def save_wav(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(np.round(np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes())


def generate(args):
    from sync.v2 import makeGuardBank
    total = int(round(args.duration * 50))
    book = codes(total, args.seed)
    guards = makeGuardBank(RATE, GUARD, 8 * args.super_spinner_size)
    t = np.arange(CORE, dtype=np.float32)
    out = np.empty((total, FRAME), dtype=np.float32)
    for n, (a, b, c) in enumerate(book):
        out[n, :CORE] = args.level * (np.sin(2 * np.pi * a * t / CORE) + np.sin(2 * np.pi * b * t / CORE) + np.sin(2 * np.pi * c * t / CORE))
        out[n, CORE:] = guards[n % len(guards)] * args.guard_amp
    save_wav(args.out, out.reshape(-1))
    print(f"TX generated: {args.out}; frames={total}, duration={total / 50:.1f}s, seed={args.seed}, guard_cycle={len(guards)}, peak={np.max(abs(out)):.3f}")


def analyze(args):
    total = int(round(args.duration * 50))
    book = codes(total, args.seed)
    lookup = {code: n for n, code in enumerate(book)}
    pcm = load_wav(args.rx)
    count = len(pcm) // FRAME
    if not count: raise ValueError("RX WAV contains no frames")
    core = pcm[:count * FRAME].reshape(-1, FRAME)[:, :CORE]
    spectrum = np.abs(np.fft.rfft(core * np.hanning(CORE), axis=1))
    band = spectrum[:, 7:53]
    ranks = np.argsort(band, axis=1)[:, ::-1]
    top3 = np.sort(ranks[:, :3] + 7, axis=1)
    peak = np.take_along_axis(band, ranks[:, :4], axis=1)
    ratio = peak[:, 2] / (peak[:, 3] + 1e-9)
    share = peak[:, :3].sum(axis=1) / (band.sum(axis=1) + 1e-9)
    rms = np.sqrt(np.mean(core * core, axis=1))
    found, rows = [], []
    for i in range(count):
        code = tuple(int(k) for k in top3[i])
        strong = rms[i] >= args.min_rms and ratio[i] >= args.min_ratio and share[i] >= args.min_share
        tx = lookup.get(code) if strong else None
        offset = tx - i if tx is not None else None
        if tx is not None: found.append((i, offset))
        rows.append((i, "" if tx is None else tx, "" if offset is None else offset, float(rms[i]), float(ratio[i]), float(share[i])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("rx_frame", "decoded_tx_frame", "offset_tx_minus_rx", "rms", "peak3_over_peak4", "spectral_share"))
        w.writerows(rows)
    print(f"RX frames={count}; confident decoded={len(found)} ({len(found)/count:.1%}); reference frames={total}")
    print("Most common offsets (TX frame - RX frame):", Counter(offset for _, offset in found).most_common(8))
    window = max(1, round(args.window * 50))
    print("\nRX time       status      offset    confirmed/window    dominance")
    for start in range(0, count, window):
        current = [offset for i, offset in found if start <= i < start + window]
        modes = Counter(current)
        offset, hits = modes.most_common(1)[0] if modes else (None, 0)
        dominance = hits / len(current) if current else 0.0
        minimum = max(5, int(0.08 * min(window, count - start)))
        status = "NO_ANCHOR" if len(current) < minimum or dominance < 0.6 else "SYNCED" if offset == 0 else "SHIFTED"
        print(f"{start / 50:6.1f}-{min(start + window, count) / 50:6.1f}s  {status:10s}  {str(offset):>6s}       {hits:3d}/{min(window,count-start):3d}         {dominance:6.1%}")
    print(f"\nPer-frame report: {args.out}")
    print("NO_ANCHOR means insufficient evidence, not proof of wrong sync.")


def main():
    p = argparse.ArgumentParser(description="Tone-coded payload + 64-state super-spinner for Telemost")
    sp = p.add_subparsers(dest="mode", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--out", default="/tmp/telemost_ss_tx.wav")
    g.add_argument("--duration", type=float, default=120)
    g.add_argument("--seed", type=int, default=20261018)
    g.add_argument("--level", type=float, default=0.10)
    g.add_argument("--guard-amp", type=float, default=0.12)
    g.add_argument("--super-spinner-size", type=int, choices=(1,8), default=8)
    a = sp.add_parser("analyze")
    a.add_argument("--rx", required=True)
    a.add_argument("--out", default="/tmp/telemost_ss_report.csv")
    a.add_argument("--duration", type=float, default=120)
    a.add_argument("--seed", type=int, default=20261018)
    a.add_argument("--window", type=float, default=5)
    a.add_argument("--min-rms", type=float, default=0.004)
    a.add_argument("--min-ratio", type=float, default=1.10)
    a.add_argument("--min-share", type=float, default=0.28)
    args = p.parse_args()
    generate(args) if args.mode == "generate" else analyze(args)

if __name__ == "__main__": main()