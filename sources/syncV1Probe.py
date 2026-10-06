import argparse
import json
import wave
from pathlib import Path

import numpy as np
from scipy.signal import firwin, lfilter

from sync.v1 import makeGuardBank


def generate(
    path,
    durationSec,
    rate=16000,
    seed=4444,
    levelDbfs=-17.0,
    guardRatio=0.8,
    spinnerSize=8
):
    if rate != 16000:
        raise ValueError("SyncV1 probe currently expects 16000 Hz")

    frameSamples = 320
    coreSamples = 240
    guardSamples = 80

    frameSec = frameSamples / rate
    frames = int(round(durationSec / frameSec))

    rng = np.random.default_rng(seed)

    white = rng.standard_normal(
        frames * coreSamples + 128
    )

    taps = firwin(
        63,
        [250, 3400],
        pass_zero=False,
        fs=rate
    )

    coreStream = lfilter(
        taps,
        [1.0],
        white
    )[128:128 + frames * coreSamples]

    cores = coreStream.reshape(
        frames,
        coreSamples
    )

    targetRms = 10.0 ** (
        levelDbfs / 20.0
    )

    cores *= targetRms / (
        np.sqrt(np.mean(cores * cores))
        + 1e-12
    )

    guards = makeGuardBank(
        rate,
        guardSamples,
        spinnerSize
    )

    out = np.zeros(
        (frames, frameSamples),
        dtype=np.float32
    )

    for frameIndex in range(frames):
        core = cores[frameIndex]

        coreRms = np.sqrt(
            np.mean(core * core)
            + 1e-12
        )

        phase = (
            frameIndex % spinnerSize
        )

        out[
            frameIndex,
            :coreSamples
        ] = core

        out[
            frameIndex,
            coreSamples:
        ] = (
            guards[phase]
            * coreRms
            * guardRatio
        )

    x = np.clip(
        out.reshape(-1),
        -0.98,
        0.98
    )

    pcm = np.round(
        x * 32767.0
    ).astype("<i2")

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())

    meta = {
        "rate": rate,
        "frames": frames,
        "duration_sec": frames * frameSec,
        "frame_samples": frameSamples,
        "core_samples": coreSamples,
        "guard_samples": guardSamples,
        "spinner_size": spinnerSize,
        "guard_ratio": guardRatio,
        "seed": seed
    }

    with open(
        path.with_suffix(".json"),
        "w"
    ) as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--seed", type=int, default=4444)
    p.add_argument("--level", type=float, default=-17.0)
    p.add_argument("--guard-ratio", type=float, default=0.8)
    p.add_argument("--spinner-size", type=int, default=8)

    args = p.parse_args()

    generate(
        args.out,
        args.duration,
        args.rate,
        args.seed,
        args.level,
        args.guard_ratio,
        args.spinner_size
    )


if __name__ == "__main__":
    main()