#!/usr/bin/env python3
"""Local ALG7 Opus test using the existing test_realtime algorithm/helpers."""
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from scipy.io import wavfile


def normalize_vectors(ps):
    shapes = {}
    for name, length in (("c_0", int(ps.G_dim_half)), ("u_0", int(ps.G_dim))):
        original = np.asarray(getattr(ps, name))
        vector = original.reshape(-1)
        if vector.size != length or not np.isfinite(vector).all():
            raise ValueError(f"Invalid {name}: expected {length} finite values")
        shapes[name] = {"before": list(original.shape), "after": list(vector.shape)}
        setattr(ps, name, vector)
    if np.any(ps.c_0 <= 0):
        raise ValueError("All c_0 radii must be positive")
    return shapes


def pcm16(values):
    return np.clip(np.rint(values), -32768, 32767).astype("<i2")


def prepare_input(source, target):
    rate, x = wavfile.read(str(source))
    if x.dtype == np.int16 and x.ndim == 1:
        shutil.copyfile(source, target)
        return
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / 2.**(8*x.dtype.itemsize-1)
    elif x.dtype == np.uint8:
        x = (x.astype(np.float64)-128.) / 128.
    elif x.dtype.kind == "f":
        x = x.astype(np.float64)
    else:
        raise ValueError("Unsupported WAV sample type")
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite mono/stereo WAV")
    if np.max(np.abs(x)) > 1.0001:
        raise ValueError("Floating WAV is outside [-1,1]; rescale it explicitly")
    wavfile.write(str(target), int(rate), pcm16(x*32768.))


def run_command(argv, out):
    argv = list(map(str, argv))
    result = subprocess.run(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    with (out/"commands.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(argv, ensure_ascii=False)+"\n"+result.stdout+"\n")
    if result.returncode:
        raise RuntimeError("Command failed; see commands.log:\n"+result.stdout[-3000:])
    return result.stdout


def opus_roundtrip(source, encoded, received, bitrate, out):
    # Same encoder options as the supplied opus_roundtrip_ffmpeg helper.
    # Keep the Ogg Opus file instead of deleting it after decoding.
    run_command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", source, "-ac", "1", "-ar", "16000", "-c:a", "libopus",
                 "-application", "voip", "-b:a", bitrate, "-vbr", "off", encoded], out)
    metadata = probe_opus(encoded, out)
    run_command(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", encoded, "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", received], out)
    return metadata


def probe_opus(encoded, out):
    metadata = json.loads(run_command([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_streams", "-show_format", "-of", "json", encoded], out))
    if not metadata.get("streams") or metadata["streams"][0]["codec_name"] != "opus":
        raise ValueError("The intermediate file is not Opus")
    return metadata


def check_received(rate, sent, got_rate, got, report):
    info = {"sent_rate": int(rate), "received_rate": int(got_rate),
            "sent_dtype": str(sent.dtype), "received_dtype": str(got.dtype),
            "sent_shape": list(sent.shape), "received_shape": list(got.shape),
            "sent_samples": len(sent), "received_samples": len(got),
            "sample_delta": len(got)-len(sent)}
    report["codec_wav"] = info
    print("OPUS WAV:", json.dumps(info), flush=True)
    if got_rate != rate or got.dtype != sent.dtype or got.ndim != 1 or not got.size:
        raise ValueError("Invalid decoded WAV format: "+json.dumps(info))
    overlap = min(len(sent), len(got))
    report["cipher_samples"] = len(sent)
    report["compared_cipher_samples"] = overlap
    changed = int(np.count_nonzero(sent[:overlap] != got[:overlap]))
    report["changed_cipher_samples_overlap"] = changed
    report["changed_cipher_samples"] = changed if len(sent) == len(got) else None


def noalign_prefix(received, got, ref, frame_size, original_len, speech_frame_size, out):
    # Evaluate only complete frames present in both streams. Never pad, shift
    # the start, or make up samples for this no-reference-alignment diagnostic.
    count = min(len(ref[0]), len(got)//frame_size)
    info = {"complete_frames_evaluated": count, "tx_frames": len(ref[0]),
            "received_samples": len(got), "evaluated_samples": count*frame_size,
            "discarded_tail_samples": len(got)-count*frame_size,
            "tx_frames_omitted": len(ref[0])-count, "start_shift": 0, "padding": 0}
    if not count:
        return None, None, 0, info
    path = received
    if len(got) != count*frame_size or count != len(ref[0]):
        path = out/"encrypted_after_opus_complete_frames.wav"
        wavfile.write(str(path), 16000, got[:count*frame_size])
    prefix_ref = (ref[0][:count], ref[1][:count], ref[2][:, :count])
    return path, prefix_ref, min(original_len, count*speech_frame_size), info


def decode_observed(core, input_wav, output_wav, raw_wav, ref, original_len, seed, out, tag):
    """Run the original decoder and repair, observing parameters before repair."""
    print("DECODE INPUT:", input_wav, "->", output_wav, flush=True)
    original_repair = core.repair_decoded_indices
    snapshots = []

    def observe(p, e, t):
        snapshots.append(tuple(np.asarray(x).copy() for x in (p, e, t)))
        return original_repair(p, e, t)

    core.repair_decoded_indices = observe
    try:
        repaired = core.decrypt_wav(input_wav, output_wav, ref, original_len, seed)
    finally:
        core.repair_decoded_indices = original_repair
    if len(snapshots) != 1:
        raise RuntimeError("Expected one repair call in the supplied decrypt_wav; inspect its current implementation")
    raw = snapshots[0]
    core.synthesize_indices(*raw, original_len, raw_wav)
    np.savez_compressed(out/(tag+"_indices.npz"),
                        raw_pitch=raw[0], raw_energy=raw[1], raw_timbre=raw[2],
                        repaired_pitch=repaired[0], repaired_energy=repaired[1],
                        repaired_timbre=repaired[2])
    return {"decoder_input": str(input_wav), "repaired_wav": str(output_wav),
            "before_repair_wav": str(raw_wav), "frames": len(raw[0])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--resume", type=Path, help="Continue a previous run from its saved ciphertext")
    parser.add_argument("--out", type=Path, default=Path("/tmp/alg7_opus_test"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bitrate")
    args = parser.parse_args(argv)
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            parser.error(executable+" is missing")
    if args.resume:
        out = args.resume.resolve()
        report_path = out/"report.json"
        required = [report_path, out/"tx_indices.npz", out/"encrypted.wav", out/"decrypted_clean.wav"]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            parser.error("Cannot resume; missing files: "+", ".join(missing))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if "seed" not in report or "bitrate" not in report:
            parser.error("The saved report lacks the original seed/bitrate")
        if ((args.seed is not None and args.seed != report["seed"])
                or (args.bitrate is not None and args.bitrate != report["bitrate"])):
            parser.error("Do not change seed/bitrate when resuming; start a new --input run")
        args.seed, args.bitrate = report["seed"], report["bitrate"]
        backup = out/"report_before_resume.json"
        if not backup.exists():
            backup.write_bytes(report_path.read_bytes())
        report.update(previous_status=report.get("status"), status="running", resumed=True)
        report.pop("error", None)
        source = None
    else:
        source = args.input.resolve()
        if not source.is_file():
            parser.error("Input WAV does not exist: "+str(source))
        args.seed = 4444 if args.seed is None else args.seed
        args.bitrate = "48k" if args.bitrate is None else args.bitrate
        args.out.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="run_", dir=str(args.out.resolve())))
        report = {"status": "running", "input": str(source), "seed": args.seed,
                  "bitrate": args.bitrate, "application": "voip", "vbr": "off"}
    print("RESULT DIRECTORY:", out, flush=True)
    try:
        # Support both test_stand/run_alg7_opus.py and algoss/alg7_kra/module.
        here = Path(__file__).resolve().parent
        root = next((p for p in (here, *here.parents)
                     if (p/"algoss/alg7_kra/test_realtime.py").is_file()), Path.cwd())
        sys.path.insert(0, str(root))
        core = importlib.import_module("algoss.alg7_kra.test_realtime")
        report["model_vector_shapes"] = normalize_vectors(core.my_ps)
        print("MODEL VECTORS:", report["model_vector_shapes"], flush=True)
        encrypted = out/"encrypted.wav"
        received = out/"encrypted_after_opus.wav"
        aligned = out/"encrypted_after_opus_aligned.wav"

        if args.resume:
            print("[1/5] Reuse existing ciphertext and TX indices", flush=True)
            with np.load(out/"tx_indices.npz") as stored:
                ref = tuple(stored[k].copy() for k in ("pitch", "energy", "timbre"))
            clean_rate, clean_audio = wavfile.read(str(out/"decrypted_clean.wav"))
            if clean_rate != 8000 or clean_audio.ndim != 1:
                raise ValueError("Expected the saved clean mono control at 8 kHz")
            original_len = len(clean_audio)
        else:
            prepared = out/"input_mono.wav"
            prepare_input(source, prepared)
            print("[1/5] Encrypt the source WAV once", flush=True)
            original_len, ref = core.encrypt_wav(prepared, encrypted, args.seed)
            np.savez_compressed(out/"tx_indices.npz", pitch=ref[0], energy=ref[1], timbre=ref[2])
        rate, sent = wavfile.read(str(encrypted))
        if rate != 16000 or sent.ndim != 1 or sent.dtype != np.int16 or not sent.size:
            raise ValueError("Expected encrypted mono PCM16 WAV at 16 kHz")
        frame_size = int(core.my_ps.flng + core.my_ps.glng)
        if len(sent) != len(ref[0])*frame_size:
            raise ValueError("Encrypted WAV length differs from the number of TX pseudo-speech frames")

        if args.resume:
            print("[2/5] Keep existing clean control unchanged", flush=True)
        else:
            print("[2/5] Decode the clean encrypted WAV as the control", flush=True)
            report["clean"] = decode_observed(core, encrypted, out/"decrypted_clean.wav",
                out/"decrypted_clean_raw.wav", ref, original_len, args.seed, out, "clean")

        print("[3/5] Encrypted WAV -> real Opus -> WAV", flush=True)
        if args.resume and (out/"encrypted.opus").is_file() and received.is_file():
            print("Reuse existing Opus file and its decoded WAV", flush=True)
            report["opus"] = probe_opus(out/"encrypted.opus", out)
        else:
            report["opus"] = opus_roundtrip(encrypted, out/"encrypted.opus", received, args.bitrate, out)
        got_rate, got = wavfile.read(str(received))
        check_received(rate, sent, got_rate, got, report)

        print("[4/5] Run the original alignment + decode + repair pipeline", flush=True)
        core.align_codec_output(encrypted, received, aligned)
        aligned_rate, aligned_audio = wavfile.read(str(aligned))
        if aligned_rate != rate or aligned_audio.dtype != sent.dtype or aligned_audio.shape != sent.shape:
            raise ValueError("Existing align_codec_output did not restore the expected WAV geometry")
        report["aligned"] = decode_observed(core, aligned, out/"decrypted_opus.wav",
            out/"decrypted_opus_aligned_raw.wav", ref, original_len, args.seed, out, "aligned")
        report["status"] = "completed"
        print("[5/5] Decode actual Opus output without reference alignment", flush=True)
        try:
            path, prefix_ref, prefix_len, info = noalign_prefix(
                received, got, ref, frame_size, original_len,
                int(core.my_analyzer.speech_frame_size), out)
            report["noalign_prefix"] = info
            print("NOALIGN COMPLETE FRAMES:", json.dumps(info), flush=True)
            if path is None:
                report["noalign"] = {"skipped": "No complete received frames"}
                report["status"] = "completed_with_noalign_skipped"
            else:
                report["noalign"] = decode_observed(core, path, out/"decrypted_opus_noalign.wav",
                    out/"decrypted_opus_noalign_raw.wav", prefix_ref, prefix_len,
                    args.seed, out, "noalign")
        except Exception as error:
            report["noalign"] = {"error": repr(error), "decoder_input": str(received)}
            report["status"] = "completed_with_noalign_failure"
            print("NOALIGN DIAGNOSTIC FAILED:", repr(error), flush=True)
        print("\nLISTEN IN THIS ORDER:", flush=True)
        for name in ("decrypted_clean_raw.wav", "decrypted_opus_noalign_raw.wav", "decrypted_opus.wav"):
            if (out/name).is_file():
                print(out/name, flush=True)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        (out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
