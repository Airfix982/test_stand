import argparse
import csv
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from sync.v2 import FixedGuardSyncV1, SyncV1Config


def floatToPcm(x):
    return np.round(
        np.clip(x, -1.0, 1.0)
        * 32767.0
    ).astype("<i2")


def main():
    p = argparse.ArgumentParser()

    src = p.add_mutually_exclusive_group(
        required=True
    )

    src.add_argument("--source")
    src.add_argument("--wav")

    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--rate", type=int, default=16000)
    p.add_argument("--spinner-size", type=int, default=8)
    p.add_argument("--super-spinner-size", type=int, default=8)
    p.add_argument("--io-samples", type=int, default=160)

    p.add_argument(
        "--acquire-threshold",
        type=float,
        default=0.28
    )

    p.add_argument(
        "--track-threshold",
        type=float,
        default=0.20
    )

    p.add_argument("--perf-interval", type=float, default=10.0, help="Log CPU/wall processing rate every N seconds of received PCM; 0 disables progress")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    wavPath = out / "recovered.wav"
    framesPath = out / "frames.csv"
    eventsPath = out / "events.jsonl"
    summaryPath = out / "summary.json"

    cfg = SyncV1Config(
        rate=args.rate,
        spinnerSize=args.spinner_size,
        superSpinnerSize=args.super_spinner_size,
        acquireThreshold=args.acquire_threshold,
        trackThreshold=args.track_threshold
    )

    wavOut = wave.open(
        str(wavPath),
        "wb"
    )

    wavOut.setnchannels(1)
    wavOut.setsampwidth(2)
    wavOut.setframerate(args.rate)

    framesFile = open(
        framesPath,
        "w",
        newline=""
    )

    framesCsv = csv.writer(framesFile)

    framesCsv.writerow([
        "frame",
        "status",
        "spinner_phase",
        "super_phase",
        "guard_code",
        "guard_pos",
        "corr",
        "timing_error_samples",
        "observed_period",
        "period_estimate",
        "predicted_frame",
        "prediction_error_frames",
        "reason"
    ])

    eventsFile = open(
        eventsPath,
        "w"
    )

    def onFrame(
        frameIndex,
        frame,
        status,
        meta
    ):
        wavOut.writeframes(
            floatToPcm(frame).tobytes()
        )

        framesCsv.writerow([
            frameIndex,
            status,
            meta.get("spinner_phase", ""),
            meta.get("super_phase", ""),
            meta.get("guard_code", ""),
            meta.get("guard_pos", ""),
            meta.get("corr", ""),
            meta.get("timing_error_samples", ""),
            meta.get("observed_period", ""),
            meta.get("period_estimate", ""),
            meta.get("predicted_frame", ""),
            meta.get("prediction_error_frames", ""),
            meta.get("reason", "")
        ])

    def onEvent(event):
        event = dict(event)
        event["monotonic"] = time.monotonic()

        eventsFile.write(
            json.dumps(event) + "\n"
        )

        eventsFile.flush()

        print(
            f'\nSYNC: {event["event"]} '
            f'state={event["state"]} '
            f'{event}'
        )

    sync = FixedGuardSyncV1(
        cfg,
        onFrame,
        onEvent
    )

    started = time.monotonic()
    perfSamples, perfWall, perfCpu, perfCalls = 0, 0.0, 0.0, []
    perfNext = args.perf_interval if args.perf_interval > 0 else float("inf")

    def measuredPush(x, timestamp):
        nonlocal perfSamples, perfWall, perfCpu, perfNext
        beforeWall, beforeCpu = time.perf_counter(), time.process_time()
        sync.push(x, timestamp)
        elapsedWall = time.perf_counter() - beforeWall
        elapsedCpu = time.process_time() - beforeCpu
        perfSamples += len(x)
        perfWall += elapsedWall
        perfCpu += elapsedCpu
        perfCalls.append(elapsedWall * 1000.0)
        audioSec = perfSamples / args.rate
        if audioSec >= perfNext:
            print(f"\nPERF pcm={audioSec:.1f}s push_cpu={perfCpu:.2f}s cpu_ratio={perfCpu/audioSec:.2f}x push_wall_ratio={perfWall/audioSec:.2f}x max_push_ms={max(perfCalls):.1f}", flush=True)
            perfNext += args.perf_interval

    proc = None

    try:
        if args.source:
            cmd = [
                "parec",
                f"--device={args.source}",
                "--format=s16le",
                f"--rate={args.rate}",
                "--channels=1",
                "--latency-msec=10"
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                bufsize=0
            )

            print("SYNC V1 RX RUNNING")

            while True:
                if (
                    args.duration > 0
                    and time.monotonic() - started
                    >= args.duration
                ):
                    break

                data = proc.stdout.read(
                    args.io_samples * 2
                )

                if not data:
                    break

                x = np.frombuffer(
                    data,
                    dtype="<i2"
                ).astype(np.float32) / 32768.0

                measuredPush(x, time.monotonic())

        else:
            with wave.open(args.wav, "rb") as w:
                if (
                    w.getnchannels() != 1
                    or w.getsampwidth() != 2
                ):
                    raise ValueError(
                        "WAV must be mono s16"
                    )

                if w.getframerate() != args.rate:
                    raise ValueError(
                        f"WAV rate={w.getframerate()}, "
                        f"expected {args.rate}"
                    )

                simulatedWall = 0.0

                while True:
                    data = w.readframes(
                        args.io_samples
                    )

                    if not data:
                        break

                    x = np.frombuffer(
                        data,
                        dtype="<i2"
                    ).astype(np.float32) / 32768.0

                    simulatedWall += (
                        len(x) / args.rate
                    )

                    measuredPush(x, simulatedWall)

                sync.finish()

    except KeyboardInterrupt:
        print("\nStopped")

    finally:
        if proc is not None:
            proc.terminate()

            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        wavOut.close()
        framesFile.close()
        eventsFile.close()

    summary = dict(sync.stats)
    audioSec = perfSamples / args.rate
    summary["performance"] = {
        "input_audio_sec": round(audioSec, 4),
        "push_wall_sec": round(perfWall, 4),
        "push_cpu_sec": round(perfCpu, 4),
        "push_wall_to_audio_ratio": round(perfWall / audioSec, 4) if audioSec else None,
        "push_cpu_to_audio_ratio": round(perfCpu / audioSec, 4) if audioSec else None,
        "max_push_ms": round(max(perfCalls), 3) if perfCalls else None,
        "p95_push_ms": round(float(np.percentile(perfCalls, 95)), 3) if perfCalls else None,
        "p99_push_ms": round(float(np.percentile(perfCalls, 99)), 3) if perfCalls else None,
        "push_calls_over_10ms": sum(ms > 10.0 for ms in perfCalls),
        "push_calls_over_20ms": sum(ms > 20.0 for ms in perfCalls),
    }

    summary.update({
        "state_final": sync.state,
        "last_frame": sync.lastFrame,
        "period_estimate": sync.periodEstimate,
        "spinner_size": cfg.spinnerSize,
        "super_spinner_size": cfg.superSpinnerSize,
        "guard_cycle": sync.guardCycle,
        "recovered_wav": str(wavPath),
        "frames_csv": str(framesPath),
        "events_jsonl": str(eventsPath)
    })

    with open(summaryPath, "w") as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()