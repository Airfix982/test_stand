"""Local ALG7 test: original KRA -> WAV -> (MP3/Opus) -> WAV -> align -> KRA.

This is only a wrapper over algoss.alg7_kra.test_realtime; it does NOT change the
KRA encrypt/decrypt implementation, pseudo-speech synthesis, or repair logic.
Compatible with Python 3.8+; requires FFmpeg, native ALG7 dependencies/models.
"""
import argparse
import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


def _require_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found (sudo apt install ffmpeg)")


def _run(command):
    print("[codec]", " ".join(map(str, command)), flush=True)
    subprocess.run([str(arg) for arg in command], check=True)


def _bitrate(value):
    value = str(value or "48k").strip().lower()
    if not re.fullmatch(r"[1-9][0-9]*k", value):
        raise ValueError("bitrate must look like '64k', '128k', '320k'")
    kbps = int(value[:-1])
    if not 6 <= kbps <= 510:
        raise ValueError("bitrate outside supported 6..510 kbit/s range")
    return value


def _resolve_codec(new_format, codec):
    """new_format is authoritative for legacy calls; explicit mismatch is an error."""
    kind = None if new_format is None else str(new_format).lower().lstrip(".")
    if kind in (None, "", "wav", "none"):
        return None
    if kind in ("ogg", "opus"):
        resolved = "opus"
    elif kind == "mp3":
        resolved = "mp3"
    else:
        raise ValueError("Unsupported new_format: %r (use None, 'mp3', 'ogg')" % new_format)
    if codec is not None and str(codec).lower() not in (resolved, "ogg" if resolved == "opus" else resolved):
        raise ValueError("codec=%r contradicts new_format=%r; change both" % (codec, new_format))
    return resolved


def codec_roundtrip(encrypted_wav, compressed_audio, decoded_raw_wav, codec, bitrate):
    """Encode encrypted WAV with actual ffmpeg codec, save bitstream, decode to PCM16 WAV.

    MP3 uses 44.1 kHz for 320k compatibility (MP3 at 16 kHz can't carry 320k).
    Opus uses 48 kHz internally. Both are decoded to 16 kHz for ALG7.
    """
    _require_ffmpeg()
    codec = str(codec).lower()
    bitrate = _bitrate(bitrate)
    encrypted_wav = Path(encrypted_wav)
    compressed_audio = Path(compressed_audio)
    decoded_raw_wav = Path(decoded_raw_wav)
    compressed_audio.parent.mkdir(parents=True, exist_ok=True)
    decoded_raw_wav.parent.mkdir(parents=True, exist_ok=True)
    base = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", encrypted_wav, "-map", "0:a:0", "-vn", "-ac", "1"]
    if codec == "mp3":
        command = base + ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", bitrate, compressed_audio]
    elif codec == "opus":
        command = base + ["-ar", "48000", "-c:a", "libopus", "-application", "voip", "-vbr", "off", "-b:a", bitrate, compressed_audio]
    else:
        raise ValueError("Unknown codec: " + codec)
    _run(command)
    _run(["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", compressed_audio,
          "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", decoded_raw_wav])
    if not compressed_audio.stat().st_size or not decoded_raw_wav.stat().st_size:
        raise RuntimeError("FFmpeg output empty")
    return compressed_audio, decoded_raw_wav


@contextmanager
def _protect_shared_encoder_temp(base_dir):
    """ALG7 speech_analyzer writes fixed BASE_DIR/temp/*. Avoid joblib collisions."""
    import fcntl
    temporary = Path(base_dir) / "temp"
    temporary.mkdir(parents=True, exist_ok=True)
    with (temporary / ".nearS_alg7_joblib.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _normalize_model_vectors(ps):
    """Only fix (1,N) vs (N,) model-shape issue; preserve every numeric value."""
    import numpy as np
    for name, length in (("c_0", int(ps.G_dim_half)), ("u_0", int(ps.G_dim))):
        value = np.asarray(getattr(ps, name)).reshape(-1)
        if value.size != length or not np.isfinite(value).all():
            raise ValueError("Invalid ALG7 model %s: shape/values" % name)
        if name == "c_0" and np.any(value <= 0):
            raise ValueError("Invalid ALG7 model c_0: radii must be positive")
        setattr(ps, name, value)


def test_7_run(org, enc, dec, errored_enc=None, org_format=None, new_format=None,
               codec=None, bitrate=None, codec_bitrate=None,
               opus_bitrate=None, mp3_bitrate=None, seed=4444, save_clean_control=True):
    """Drop-in for process_one_file(); all artifacts are saved alongside enc and dec.

    new_format=None -> clean, 'mp3' -> MP3, 'ogg'/'opus' -> Ogg/Opus.
    codec and new_format must agree when both specified.
    """
    if errored_enc is not None:
        raise ValueError("errored_enc is not used by this pipeline; use new_format instead")
    if org_format is not None and str(org_format).lower().lstrip(".") != "wav":
        raise ValueError("Only WAV source supported by original ALG7 test_realtime.py")
    enc = Path(enc).expanduser().resolve()
    dec = Path(dec).expanduser().resolve()
    org = Path(org).expanduser().resolve()
    if not org.is_file():
        raise FileNotFoundError(str(org))
    if enc == dec or org in (enc, dec):
        raise ValueError("org, enc and dec must be three different file paths")
    if enc.suffix.lower() != '.wav' or dec.suffix.lower() != '.wav':
        raise ValueError("ALG7 original encrypted and decrypted paths must end in .wav")
    enc.parent.mkdir(parents=True, exist_ok=True)
    dec.parent.mkdir(parents=True, exist_ok=True)
    mode = _resolve_codec(new_format, codec)
    selected_bitrate = (mp3_bitrate if mode == "mp3" else opus_bitrate if mode == "opus" else None)
    selected_bitrate = selected_bitrate or codec_bitrate or bitrate or ("64k" if mode == "mp3" else "48k")
    if mode:
        selected_bitrate = _bitrate(selected_bitrate)
    # IMPORTANT: import core lazily so ffmpeg/--help work without native models.
    from . import test_realtime as core
    _normalize_model_vectors(core.my_ps)
    report = {
        "org": str(org), "enc": str(enc), "dec": str(dec),
        "seed": int(seed), "codec": mode or "none", "bitrate": selected_bitrate if mode else None,
        "files": {"original": str(org), "encrypted_pcm16": str(enc)},
        "status": "running", "alignment_uses_clean_tx_reference": bool(mode),
    }
    report_path = enc.with_name(enc.stem + ".alg7_run.json")
    try:
        with _protect_shared_encoder_temp(core.BASE_DIR):
            original_len, ref = core.encrypt_wav(str(org), str(enc), seed=int(seed))
        report["frames"] = int(len(ref[0]))
        report["original_8k_samples"] = int(original_len)
        # Original indices are useful for offline error analysis, no need to re-encrypt.
        import numpy as np
        indices_path = enc.with_name(enc.stem + ".tx_indices.npz")
        np.savez_compressed(str(indices_path), pitch=ref[0], energy=ref[1], timbre=ref[2])
        report["files"]["tx_indices"] = str(indices_path)
        if mode is None:
            core.decrypt_wav(str(enc), str(dec), ref, original_len, seed=int(seed))
        else:
            ext = ".mp3" if mode == "mp3" else ".ogg"
            compressed_path = enc.with_suffix(enc.suffix + ext)  # enc1.wav.mp3 / enc1.wav.ogg
            raw_path = enc.with_name(enc.stem + "." + mode + "_raw.wav")
            aligned_path = enc.with_name(enc.stem + "." + mode + "_aligned.wav")
            report["files"].update(codec_file=str(compressed_path),
                                   decoded_codec_raw_wav=str(raw_path),
                                   decoded_codec_aligned_wav=str(aligned_path))
            codec_roundtrip(enc, compressed_path, raw_path, mode, selected_bitrate)
            # IMPORTANT: pass decoded CODEC WAV, not clean TX `enc` to decrypt().
            core.align_codec_output(str(enc), str(raw_path), str(aligned_path))
            core.decrypt_wav(str(aligned_path), str(dec), ref, original_len, seed=int(seed))
            if save_clean_control:
                clean_path = dec.with_name(dec.stem + "_clean.wav")
                report["files"]["clean_control_wav"] = str(clean_path)
                core.decrypt_wav(str(enc), str(clean_path), ref, original_len, seed=int(seed))
        report["files"]["decrypted_wav"] = str(dec)
        report["status"] = "completed"
        print("[ALG7] completed:", json.dumps(report["files"], ensure_ascii=False, indent=2), flush=True)
        return report
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Local ALG7 clean/MP3/Ogg-Opus pipeline; all intermediate files retained")
    ap.add_argument("--input", required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--dec", required=True)
    ap.add_argument("--codec", choices=("none", "mp3", "opus"), default="none")
    ap.add_argument("--bitrate", default="64k")
    ap.add_argument("--seed", type=int, default=4444)
    ap.add_argument("--no-clean-control", action="store_true")
    a = ap.parse_args()
    test_7_run(a.input, a.enc, a.dec, org_format="wav",
               new_format={"none": None, "mp3": "mp3", "opus": "ogg"}[a.codec],
               codec=a.codec, bitrate=a.bitrate, seed=a.seed,
               save_clean_control=not a.no_clean_control)


if __name__ == "__main__":
    main()
