import argparse, csv, json, wave
from pathlib import Path

import numpy as np
from scipy.signal import correlate

from sources.probe import DeterministicProbe


def readWavMono(path):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16bit WAV supported")

        rate, channels = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)

    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)

    return rate, x


def normalizedCorrelationScore(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.sqrt(np.dot(a, a) * np.dot(b, b)) + 1e-12
    return float(np.dot(a, b) / den)


def bestOffset(reference, segment):
    ref = reference.astype(np.float64)
    seg = segment.astype(np.float64)
    n = len(ref)

    if len(seg) < n:
        return None, None

    ref -= np.mean(ref)
    refEnergy = np.dot(ref, ref) + 1e-12

    corr = correlate(seg, ref, mode="valid", method="fft")

    ones = np.ones(n, dtype=np.float64)
    winSum = np.convolve(seg, ones, mode="valid")
    winSumSq = np.convolve(seg * seg, ones, mode="valid")
    winEnergy = np.maximum(winSumSq - (winSum * winSum) / n, 1e-12)

    score = corr / np.sqrt(refEnergy * winEnergy)
    i = int(np.argmax(score))

    return i, float(score[i])


def distortionMetrics(reference, received):
    ref = reference.astype(np.float64)
    got = received.astype(np.float64)

    rmsRef = float(np.sqrt(np.mean(ref * ref)))
    rmsRx = float(np.sqrt(np.mean(got * got)))
    rmsRatio = rmsRx / max(rmsRef, 1e-12)
    rawRmse = float(np.sqrt(np.mean((got - ref) ** 2)))

    refMean, gotMean = float(np.mean(ref)), float(np.mean(got))
    ref0, got0 = ref - refMean, got - gotMean

    gain = float(np.dot(ref0, got0) / (np.dot(ref0, ref0) + 1e-12))
    dcOffset = gotMean - gain * refMean

    error = got - (gain * ref + dcOffset)
    gainRmse = float(np.sqrt(np.mean(error * error)))
    signalRms = float(np.sqrt(np.mean(got0 * got0)))

    gainNrmse = gainRmse / max(signalRms, 1e-12)
    gainSnrDb = float(20.0 * np.log10(max(signalRms, 1e-12) / max(gainRmse, 1e-12)))

    return rmsRef, rmsRx, rmsRatio, rawRmse, gain, dcOffset, gainRmse, gainNrmse, gainSnrDb


def determineBlockCount(rate, blockSamples, rx, txPath=None, txDuration=None, maxBlocks=None):
    if txPath:
        txRate, tx = readWavMono(txPath)
        if txRate != rate:
            raise ValueError(f"TX/RX rates differ: {txRate} != {rate}")
        count = len(tx) // blockSamples
    elif txDuration is not None:
        count = int(np.floor(float(txDuration) * rate / blockSamples + 1e-9))
    elif maxBlocks is not None:
        count = int(maxBlocks)
    else:
        count = len(rx) // blockSamples

    if maxBlocks is not None:
        count = min(count, int(maxBlocks))

    return max(0, count)


def findPersistentTimingEvents(rows, rate, thresholdMs=5.0, window=3, corrMin=0.2):
    result = []
    threshold = thresholdMs * rate / 1000.0

    for i in range(window, len(rows) - window + 1):
        instantDelta = rows[i]["lag_samples"] - rows[i - 1]["lag_samples"]

        if abs(instantDelta) < threshold:
            continue

        before = rows[i - window:i]
        after = rows[i:i + window]

        if len(after) < window:
            continue

        if min(r["corr"] for r in before + after) < corrMin:
            continue

        oldLag = float(np.median([r["lag_samples"] for r in before]))
        newLag = float(np.median([r["lag_samples"] for r in after]))
        delta = newLag - oldLag

        if abs(delta) < threshold or delta * instantDelta <= 0:
            continue

        result.append({
            "block": rows[i]["block"],
            "tx_time_s": rows[i]["tx_time_s"],
            "before_lag_samples": oldLag,
            "after_lag_samples": newLag,
            "delta_samples": delta,
            "delta_ms": 1000.0 * delta / rate,
            "instant_delta_ms": 1000.0 * instantDelta / rate,
            "corr_before_min": min(r["corr"] for r in before),
            "corr_after_min": min(r["corr"] for r in after)
        })

    return result


def analyze(rxPath, outDir, seed=4444, blockSec=1.0, dbfsLvl=-15.0,
            initialSearch=30.0, searchMs=150.0, txPath=None, txDuration=None,
            maxBlocks=None, eventMs=2.0, eventCorrMin=0.20, referenceBlocks=5,
            persistentEventMs=5.0, persistentWindow=3):

    rate, rx = readWavMono(rxPath)
    probe = DeterministicProbe(rate, seed, blockSec, dbfsLvl)

    n = getattr(probe, "blockSamples", None)
    if n is None:
        n = getattr(probe, "block_samples", None)
    if n is None:
        raise AttributeError("DeterministicProbe has no blockSamples")

    out = Path(outDir)
    out.mkdir(parents=True, exist_ok=True)

    totalBlocks = determineBlockCount(
        rate, n, rx, txPath=txPath,
        txDuration=txDuration, maxBlocks=maxBlocks
    )

    if totalBlocks == 0:
        raise ValueError("zero TX blocks to analyze")

    ref0 = probe.block(0).astype(np.float64)
    initialEnd = min(len(rx), int(initialSearch * rate) + n)

    off, acquisitionCorr = bestOffset(ref0, rx[:initialEnd])
    if off is None:
        raise RuntimeError("initial acquisition failed")

    predicted = off
    radius = int(searchMs * rate / 1000.0)
    rows = []

    for i in range(totalBlocks):
        ref = probe.block(i).astype(np.float64)

        lo = max(0, predicted - radius)
        hi = min(len(rx), predicted + radius + n)

        if hi - lo < n:
            break

        rel, _ = bestOffset(ref, rx[lo:hi])
        if rel is None:
            break

        pos = lo + rel
        got = rx[pos:pos + n]

        if len(got) < n:
            break

        corr = normalizedCorrelationScore(ref, got)
        metrics = distortionMetrics(ref, got)
        rmsRef, rmsRx, rmsRatio, rawRmse, gain, dcOffset, gainRmse, gainNrmse, gainSnrDb = metrics

        lag = int(pos - i * n)

        rows.append({
            "block": i, "tx_time_s": i * blockSec,
            "tx_sample": i * n, "rx_sample": pos,
            "lag_samples": lag, "lag_ms": 1000.0 * lag / rate,
            "corr": corr, "rmse": rawRmse,
            "rms_tx": rmsRef, "rms_rx": rmsRx, "rms_ratio": rmsRatio,
            "gain_ls": gain, "dc_offset": dcOffset,
            "gain_rmse": gainRmse, "gain_nrmse": gainNrmse,
            "gain_snr_db": gainSnrDb
        })

        predicted = pos + n

    if not rows:
        raise RuntimeError("no blocks analyzed")

    reliableRows = [r for r in rows if r["corr"] >= eventCorrMin]
    refPool = reliableRows[:max(1, int(referenceBlocks))]
    referenceLag = float(np.median([r["lag_samples"] for r in refPool])) if refPool else float(rows[0]["lag_samples"])

    thresholdSamples = eventMs * rate / 1000.0
    events, prev = [], None

    for r in rows:
        r["relative_lag_samples"] = r["lag_samples"] - referenceLag
        r["relative_lag_ms"] = 1000.0 * r["relative_lag_samples"] / rate
        r["reliable"] = int(r["corr"] >= eventCorrMin)

        if prev is None:
            r["lag_delta_samples"] = 0.0
            r["lag_delta_ms"] = 0.0
        else:
            delta = r["lag_samples"] - prev["lag_samples"]
            r["lag_delta_samples"] = delta
            r["lag_delta_ms"] = 1000.0 * delta / rate

            if abs(delta) >= thresholdSamples:
                events.append({
                    "block": r["block"], "tx_time_s": r["tx_time_s"],
                    "prev_lag_samples": prev["lag_samples"], "lag_samples": r["lag_samples"],
                    "delta_samples": delta, "delta_ms": 1000.0 * delta / rate,
                    "corr_prev": prev["corr"], "corr": r["corr"],
                    "reliable": int(prev["corr"] >= eventCorrMin and r["corr"] >= eventCorrMin)
                })

        prev = r

    persistentEvents = findPersistentTimingEvents(
        rows, rate, thresholdMs=persistentEventMs,
        window=persistentWindow, corrMin=eventCorrMin
    )

    alignmentFields = [
        "block", "tx_time_s", "tx_sample", "rx_sample",
        "lag_samples", "lag_ms", "relative_lag_samples", "relative_lag_ms",
        "lag_delta_samples", "lag_delta_ms", "corr", "reliable", "rmse",
        "rms_tx", "rms_rx", "rms_ratio", "gain_ls", "dc_offset",
        "gain_rmse", "gain_nrmse", "gain_snr_db"
    ]

    alignmentPath = out / "alignment.csv"
    with alignmentPath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=alignmentFields)
        writer.writeheader()
        writer.writerows(rows)

    eventFields = [
        "block", "tx_time_s", "prev_lag_samples", "lag_samples",
        "delta_samples", "delta_ms", "corr_prev", "corr", "reliable"
    ]

    eventPath = out / "timing_events.csv"
    with eventPath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=eventFields)
        writer.writeheader()
        writer.writerows(events)

    persistentFields = [
        "block", "tx_time_s", "before_lag_samples", "after_lag_samples",
        "delta_samples", "delta_ms", "instant_delta_ms",
        "corr_before_min", "corr_after_min"
    ]

    persistentPath = out / "persistent_timing_events.csv"
    with persistentPath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=persistentFields)
        writer.writeheader()
        writer.writerows(persistentEvents)

    statsRows = reliableRows if reliableRows else rows
    lags = np.array([r["lag_samples"] for r in statsRows], dtype=float)
    corrs = np.array([r["corr"] for r in rows], dtype=float)
    reliableEvents = [e for e in events if e["reliable"]]

    summary = {
        "rate": rate,
        "tx_blocks_requested": totalBlocks,
        "blocks_analyzed": len(rows),
        "blocks_reliable": len(reliableRows),
        "reliable_corr_threshold": eventCorrMin,
        "acquisition_corr": acquisitionCorr,

        "initial_lag_samples": float(rows[0]["lag_samples"]),
        "initial_lag_ms": float(rows[0]["lag_ms"]),
        "reference_lag_samples": referenceLag,
        "reference_lag_ms": 1000.0 * referenceLag / rate,

        "lag_min_reliable": float(lags.min()),
        "lag_max_reliable": float(lags.max()),
        "lag_span_samples_reliable": float(lags.max() - lags.min()),
        "lag_span_ms_reliable": float(1000.0 * (lags.max() - lags.min()) / rate),
        "final_relative_lag_ms": float(statsRows[-1]["relative_lag_ms"]),

        "corr_mean": float(corrs.mean()),
        "corr_min": float(corrs.min()),
        "corr_median": float(np.median(corrs)),

        "rms_ratio_median_reliable": float(np.median([r["rms_ratio"] for r in statsRows])),
        "gain_rmse_mean_reliable": float(np.mean([r["gain_rmse"] for r in statsRows])),
        "gain_nrmse_mean_reliable": float(np.mean([r["gain_nrmse"] for r in statsRows])),
        "gain_snr_db_mean_reliable": float(np.mean([r["gain_snr_db"] for r in statsRows])),

        "timing_event_threshold_ms": eventMs,
        "timing_events_total": len(events),
        "timing_events_reliable": len(reliableEvents),
        "max_abs_timing_jump_ms": max((abs(e["delta_ms"]) for e in reliableEvents), default=0.0),

        "persistent_event_threshold_ms": persistentEventMs,
        "persistent_event_window_blocks": persistentWindow,
        "persistent_timing_events": len(persistentEvents),
        "persistent_max_abs_jump_ms": max((abs(e["delta_ms"]) for e in persistentEvents), default=0.0),

        "output_dir": str(out.resolve()),
        "alignment_csv": str(alignmentPath.resolve()),
        "timing_events_csv": str(eventPath.resolve()),
        "persistent_timing_events_csv": str(persistentPath.resolve())
    }

    if txPath is None and txDuration is None and maxBlocks is None:
        summary["warning"] = "TX length unknown; RX tail may be analyzed as fake probe blocks"

    summaryPath = out / "summary.json"
    summary["summary_json"] = str(summaryPath.resolve())
    summaryPath.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary, alignmentPath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rx", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tx", default=None)
    parser.add_argument("--tx-duration", type=float, default=None)
    parser.add_argument("--seed", type=int, default=4444)
    parser.add_argument("--block-seconds", type=float, default=1.0)
    parser.add_argument("--level-dbfs", type=float, default=-15.0)
    parser.add_argument("--initial-search-s", type=float, default=30.0)
    parser.add_argument("--search-ms", type=float, default=150.0)
    parser.add_argument("--event-ms", type=float, default=2.0)
    parser.add_argument("--event-corr-min", type=float, default=0.20)
    parser.add_argument("--reference-blocks", type=int, default=5)

    args = parser.parse_args()

    summary, _ = analyze(
        rxPath=args.rx, outDir=args.out, seed=args.seed,
        blockSec=args.block_seconds, dbfsLvl=args.level_dbfs,
        initialSearch=args.initial_search_s, searchMs=args.search_ms,
        txPath=args.tx, txDuration=args.tx_duration,
        eventMs=args.event_ms, eventCorrMin=args.event_corr_min,
        referenceBlocks=args.reference_blocks
    )

    print(json.dumps(summary, indent=2))
    print(f"\nRESULTS: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()