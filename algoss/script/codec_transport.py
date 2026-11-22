
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile


def codec_roundtrip_inplace(enc, new_format, bitrate=None):
    """Save original ciphertext; replace enc.wav with codec-decoded WAV."""
    enc = Path(enc)
    original = enc.with_name("org_" + enc.name)

    if new_format is None:
        original.unlink(missing_ok=True)
        return

    fmt = str(new_format).lower().lstrip(".")
    if fmt not in ("mp3", "ogg", "opus"):
        raise ValueError(f"Unsupported codec format: {new_format}")

    opus = fmt in ("ogg", "opus")

    fs, data = wavfile.read(enc, mmap=True)
    if data.ndim != 1:
        raise ValueError("Codec transport expects mono WAV")

    pcm = {
        np.dtype("uint8"): "pcm_u8",
        np.dtype("int16"): "pcm_s16le",
        np.dtype("int32"): "pcm_s32le",
        np.dtype("float32"): "pcm_f32le",
    }.get(data.dtype)

    if pcm is None:
        raise ValueError(f"Unsupported encrypted WAV dtype: {data.dtype}")

    expected_len = len(data)
    original_dtype = data.dtype
    del data

    # Move clean ciphertext to org_encN.wav.
    enc.replace(original)

    compressed = Path(str(enc) + (".ogg" if opus else ".mp3"))
    options = (
        ["-ar", "48000", "-c:a", "libopus", "-application", "audio"]
        if opus else
        ["-ar", "44100", "-c:a", "libmp3lame"]
    )
    br = str(bitrate or ("96k" if opus else "128k"))

    base = [
        "ffmpeg", "-y", "-nostdin",
        "-hide_banner", "-loglevel", "error"
    ]

    subprocess.run(base + [
        "-i", str(original), "-map", "0:a:0", "-vn", "-ac", "1",
        *options, "-b:a", br, str(compressed)
    ], check=True)

    subprocess.run(base + [
        "-i", str(compressed), "-map", "0:a:0", "-vn", "-ac", "1",
        "-ar", str(fs), "-c:a", pcm, str(enc)
    ], check=True)

    actual_fs, decoded = wavfile.read(enc)

    if actual_fs != fs or decoded.dtype != original_dtype:
        raise ValueError("Codec did not restore WAV rate/dtype")

    
    delta = len(decoded) - expected_len

    # Допускаем небольшую погрешность длительности
    # после ресемплинга и MP3/Opus (не более 5 мс).
    max_delta = max(32, int(round(fs * 0.005)))

    if abs(delta) > max_delta:
        raise ValueError(
            f"Codec changed length too much: "
            f"{expected_len} -> {len(decoded)}, "
            f"limit={max_delta}"
        )

    if delta > 0:
        # Лишние отсчёты в конце.
        decoded = decoded[:expected_len]

    elif delta < 0:
        # Дополняем хвост тишиной.
        silence = 128 if decoded.dtype == np.uint8 else 0
        decoded = np.pad(
            decoded,
            (0, -delta),
            mode="constant",
            constant_values=silence
        )

    if delta:
        print(
            f"Codec length aligned: {delta:+d} samples",
            flush=True
        )

    wavfile.write(enc, fs, decoded)


    print(
        f"CODEC {fmt}: {original.name} -> "
        f"{compressed.name} -> {enc.name}",
        flush=True
    )
