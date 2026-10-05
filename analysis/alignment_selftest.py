import csv, shutil, wave
from pathlib import Path

import numpy as np

from analysis.alignment import analyze
from sources.probe import DeterministicProbe


RATE = 44100
SEED = 4444
DURATION = 120
BLOCK_SEC = 1.0
OUTPUT = Path("alignment_selftest_output").resolve()


def writeWav(path, data):
    x = np.clip(data, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(x.tobytes())


def generateTx():
    probe = DeterministicProbe(RATE, SEED, BLOCK_SEC, -15.0)
    return np.concatenate([probe.block(i) for i in range(DURATION)]).astype(np.float64)


def applyTimelineEvent(x, sourceSecond, deltaMs, initialDelay, accumulatedShift):
    count = int(round(abs(deltaMs) * RATE / 1000.0))
    pos = initialDelay + int(sourceSecond * RATE) + accumulatedShift

    if deltaMs < 0:
        x = np.concatenate([x[:pos], x[pos + count:]])
        accumulatedShift -= count
    else:
        start = max(0, pos - count)
        filler = x[start:pos]

        if len(filler) < count:
            filler = np.pad(filler, (count - len(filler), 0))

        x = np.concatenate([x[:pos], filler, x[pos:]])
        accumulatedShift += count

    return x, accumulatedShift


def main():
    shutil.rmtree(OUTPUT, ignore_errors=True)
    analysisDir = OUTPUT / "analysis"
    analysisDir.mkdir(parents=True)

    tx = generateTx()
    initialDelay = 3 * RATE

    rx = np.concatenate([np.zeros(initialDelay), tx])
    shift = 0

    rx, shift = applyTimelineEvent(rx, 25, -30.0, initialDelay, shift)
    rx, shift = applyTimelineEvent(rx, 55, +20.0, initialDelay, shift)
    rx, shift = applyTimelineEvent(rx, 85, -10.0, initialDelay, shift)

    rng = np.random.default_rng(123)
    rx = rx * 0.18 + rng.normal(0, 40, len(rx))

    txPath = OUTPUT / "tx.wav"
    rxPath = OUTPUT / "rx.wav"

    writeWav(txPath, tx)
    writeWav(rxPath, rx)

    analyze(
        rxPath=rxPath, outDir=analysisDir,
        txPath=txPath, seed=SEED,
        eventMs=2.0, persistentEventMs=5.0,
        persistentWindow=3
    )

    persistentPath = analysisDir / "persistent_timing_events.csv"

    with persistentPath.open(encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    expected = {25: -30.0, 55: 20.0, 85: -10.0}
    detected = {int(e["block"]): float(e["delta_ms"]) for e in events}

    print("\nEXPECTED:")
    for block, delta in expected.items():
        print(f"  block {block}: {delta:+.1f} ms")

    print("\nDETECTED:")
    for e in events:
        print(f"  block {e['block']}: {float(e['delta_ms']):+.3f} ms")

    errors = []
    for block, delta in expected.items():
        if block not in detected:
            errors.append(f"missing block {block}")
        elif abs(detected[block] - delta) > 0.1:
            errors.append(f"block {block}: expected {delta}, got {detected[block]}")

    unexpected = set(detected) - set(expected)
    if unexpected:
        errors.append(f"unexpected events: {sorted(unexpected)}")

    print("\nSELF-TEST:", "PASS" if not errors else "FAIL")

    if errors:
        for error in errors:
            print(" ", error)

    print(f"\nOUTPUT: {OUTPUT}")
    print(f"SUMMARY: {analysisDir / 'summary.json'}")
    print(f"ALIGNMENT: {analysisDir / 'alignment.csv'}")
    print(f"EVENTS: {analysisDir / 'timing_events.csv'}")
    print(f"PERSISTENT: {persistentPath}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()