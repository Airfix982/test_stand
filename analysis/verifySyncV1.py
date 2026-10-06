import argparse
import json
import wave

import numpy as np


FRAME = 320
CORE = 240


def readWav(path):
    with wave.open(path, "rb") as w:
        if (
            w.getnchannels() != 1
            or w.getsampwidth() != 2
        ):
            raise ValueError("mono s16 required")

        rate = w.getframerate()

        x = np.frombuffer(
            w.readframes(w.getnframes()),
            dtype="<i2"
        ).astype(np.float32)

    return rate, x


def corr(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-9 or nb < 1e-9:
        return 0.0

    return float(
        np.dot(a, b)
        / (na * nb)
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--tx", required=True)
    p.add_argument("--rx", required=True)
    p.add_argument("--radius", type=int, default=4)

    args = p.parse_args()

    txRate, tx = readWav(args.tx)
    rxRate, rx = readWav(args.rx)

    if txRate != rxRate:
        raise ValueError("sample rates differ")

    txFrames = len(tx) // FRAME
    rxFrames = len(rx) // FRAME

    total = min(
        txFrames,
        rxFrames
    )

    lost = 0
    checked = 0
    indexErrors = []

    sameCorrs = []

    for i in range(total):
        rxCore = rx[
            i * FRAME:
            i * FRAME + CORE
        ]

        if np.sqrt(
            np.mean(rxCore * rxCore)
        ) < 1.0:
            lost += 1
            continue

        checked += 1

        bestFrame = None
        bestCorr = -2.0

        lo = max(
            0,
            i - args.radius
        )

        hi = min(
            txFrames - 1,
            i + args.radius
        )

        for j in range(lo, hi + 1):
            txCore = tx[
                j * FRAME:
                j * FRAME + CORE
            ]

            c = corr(
                rxCore,
                txCore
            )

            if c > bestCorr:
                bestCorr = c
                bestFrame = j

        txSame = tx[
            i * FRAME:
            i * FRAME + CORE
        ]

        same = corr(
            rxCore,
            txSame
        )

        sameCorrs.append(same)

        if bestFrame != i:
            indexErrors.append({
                "rx_frame": i,
                "best_tx_frame": bestFrame,
                "delta_frames": bestFrame - i,
                "best_corr": bestCorr,
                "same_index_corr": same
            })

    result = {
        "tx_frames": txFrames,
        "rx_frames": rxFrames,
        "frames_checked": checked,
        "frames_lost": lost,
        "index_errors": len(indexErrors),
        "max_abs_index_error": (
            max(
                abs(e["delta_frames"])
                for e in indexErrors
            )
            if indexErrors
            else 0
        ),
        "mean_same_index_corr": (
            float(np.mean(sameCorrs))
            if sameCorrs
            else 0.0
        ),
        "first_index_errors": (
            indexErrors[:20]
        )
    }

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__ == "__main__":
    main()