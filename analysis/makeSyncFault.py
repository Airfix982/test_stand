import argparse
import wave

import numpy as np


def readWav(path):
    with wave.open(path, "rb") as w:
        rate = w.getframerate()

        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("mono s16 WAV required")

        x = np.frombuffer(
            w.readframes(w.getnframes()),
            dtype="<i2"
        ).copy()

    return rate, x


def writeWav(path, rate, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(x, dtype="<i2").tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--type", choices=["dropout", "delete", "insert"], required=True)
    p.add_argument("--at", type=float, required=True)
    p.add_argument("--ms", type=float, required=True)
    args = p.parse_args()

    rate, x = readWav(args.input)

    pos = int(round(args.at * rate))
    n = int(round(args.ms * rate / 1000.0))

    if args.type == "dropout":
        y = x.copy()
        y[pos:pos + n] = 0

    elif args.type == "delete":
        y = np.concatenate((x[:pos], x[pos + n:]))

    else:
        begin = max(0, pos - n)
        inserted = x[begin:pos]
        y = np.concatenate((x[:pos], inserted, x[pos:]))

    writeWav(args.out, rate, y)

    print(
        f"type={args.type} at={args.at}s ms={args.ms} "
        f"samples={n} old={len(x)} new={len(y)}"
    )


if __name__ == "__main__":
    main()