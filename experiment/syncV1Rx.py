import argparse
import csv
import json
import subprocess
import time
import wave
from pathlib import Path

import numpy as np

from sync.v1 import FixedGuardSyncV1, SyncV1Config


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

                sync.push(
                    x,
                    time.monotonic()
                )

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

                    sync.push(
                        x,
                        simulatedWall
                    )

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

    summary.update({
        "state_final": sync.state,
        "last_frame": sync.lastFrame,
        "period_estimate": sync.periodEstimate,
        "spinner_size": cfg.spinnerSize,
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