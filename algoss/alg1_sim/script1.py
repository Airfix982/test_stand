import wave
import math
import numpy as np
from numba import njit
import script.utils as utils
import importlib



importlib.reload(utils)
me = utils


def read_wav_mono(path):
    with wave.open(path, "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        n_frames = f.getnframes()
        comptype = f.getcomptype()
        raw = f.readframes(n_frames)

    if comptype != "NONE":
        raise ValueError("only uncompressed PCM wav supported")
    if n_channels != 1:
        raise ValueError("mono only")

    bits = sampwidth * 8

    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
        min_val, max_val = 0, 255
        dtype_name = "uint8"
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.int64)
        min_val, max_val = -32768, 32767
        dtype_name = "int16"
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.int64)
        min_val, max_val = -(1 << 31), (1 << 31) - 1
        dtype_name = "int32"
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    meta = {
        "sample_rate": framerate,
        "sample_width_bytes": sampwidth,
        "bits_per_sample": bits,
        "min_value": min_val,
        "max_value": max_val,
        "dtype_name": dtype_name,
    }
    return arr, meta


def write_wav_mono(path, samples, meta):
    samples = clip_to_pcm_range(samples, meta)

    sampwidth = meta["sample_width_bytes"]
    framerate = meta["sample_rate"]

    if sampwidth == 1:
        raw = samples.astype(np.uint8).tobytes()
    elif sampwidth == 2:
        raw = samples.astype("<i2").tobytes()
    elif sampwidth == 4:
        raw = samples.astype("<i4").tobytes()
    else:
        raise ValueError(f"unsupported sample width: {sampwidth} bytes")

    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(raw)


def clip_to_pcm_range(samples, meta):
    minimum = meta["min_value"]
    maximum = meta["max_value"]
    return np.clip(np.rint(samples), minimum, maximum).astype(np.int64)

def signed_to_unsigned_words(values, bits):
    mask = (1 << bits) - 1
    return (values.astype(np.int64) & mask).astype(np.uint64)


def unsigned_to_signed_words(values, bits):
    sign_bit = 1 << (bits - 1)
    full = 1 << bits
    v = values.astype(np.int64)
    return np.where(v >= sign_bit, v - full, v).astype(np.int64)


@njit
def trajectory(x0: float, r: float, num_points: int):
    points = np.empty(num_points, dtype=np.float64)
    x = x0
    for i in range(num_points):
        x = r * math.sin(math.pi * x) + r * ((1000.0 * x) % 1.0)
        points[i] = x
    return points


@njit
def random_bits(traj):
    bits = np.empty(len(traj), dtype=np.uint8)
    for i in range(len(traj)):
        bits[i] = 1 if ((1e10 * traj[i]) % 1.0) > 0.5 else 0
    return bits


def random_bits_to_words(bits: np.ndarray, bits_per_sample: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) % bits_per_sample != 0:
        raise ValueError("bits length must be divisible by bits_per_sample")

    bits = bits.reshape(-1, bits_per_sample)

    powers = (1 << np.arange(bits_per_sample - 1, -1, -1)).astype(np.uint64)
    values = bits.dot(powers).astype(np.uint64)

    return values


def generate_key_stream_words(num_samples: int, x0: float, r: float, bits_per_sample: int):
    total_bits = bits_per_sample * num_samples
    traj = trajectory(x0, r, total_bits)
    next_x0 = traj[-1] if len(traj) > 0 else x0
    bits = random_bits(traj)
    key_words = random_bits_to_words(bits, bits_per_sample)
    return key_words[:num_samples], next_x0


def split_into_chunks(samples, chunk_size):
    return [samples[i:i + chunk_size] for i in range(0, len(samples), chunk_size)]


def xor_chunk_with_keystream(chunk_samples, key_words, meta):
    bits = meta["bits_per_sample"]

    if bits == 8:
        chunk_u = chunk_samples.astype(np.uint64)
        out_u = np.bitwise_xor(chunk_u, key_words.astype(np.uint64))
        return np.clip(out_u, 0, 255).astype(np.int64)

    chunk_u = signed_to_unsigned_words(chunk_samples, bits)
    out_u = np.bitwise_xor(chunk_u, key_words.astype(np.uint64))
    out_s = unsigned_to_signed_words(out_u, bits)
    return out_s.astype(np.int64)


def encrypt_signal(samples, meta, x0, r, chunk_size):
    chunks = split_into_chunks(samples, chunk_size)
    enc_chunks = []

    bits = meta["bits_per_sample"]

    for chunk in chunks:
        key_words, x0 = generate_key_stream_words(
            num_samples=len(chunk),
            x0=x0,
            r=r,
            bits_per_sample=bits,
        )
        enc_chunk = xor_chunk_with_keystream(chunk, key_words, meta)
        enc_chunks.append(enc_chunk)

    encrypted = np.concatenate(enc_chunks).astype(np.int64) if enc_chunks else np.array([], dtype=np.int64)
    return encrypted, x0


def decrypt_signal(samples, meta, x0, r, chunk_size):
    chunks = split_into_chunks(samples, chunk_size)
    dec_chunks = []

    bits = meta["bits_per_sample"]

    for chunk in chunks:
        key_words, x0 = generate_key_stream_words(
            num_samples=len(chunk),
            x0=x0,
            r=r,
            bits_per_sample=bits,
        )
        dec_chunk = xor_chunk_with_keystream(chunk, key_words, meta)
        dec_chunks.append(dec_chunk)

    decrypted = np.concatenate(dec_chunks).astype(np.int64) if dec_chunks else np.array([], dtype=np.int64)
    return decrypted, x0


def encrypt_wav_file_realtime_like(
    input_wav_path: str,
    output_wav_path: str,
    x0: float,
    r: float,
    block_samples: int
) -> float:
    samples, meta = read_wav_mono(input_wav_path)

    encrypted, x0 = encrypt_signal(
        samples=samples,
        meta=meta,
        x0=x0,
        r=r,
        chunk_size=block_samples
    )

    write_wav_mono(output_wav_path, encrypted, meta)
    return x0


def decrypt_wav_file_realtime_like(
    input_wav_path: str,
    output_wav_path: str,
    x0: float,
    r: float,
    block_samples: int
) -> float:
    encrypted, meta = read_wav_mono(input_wav_path)

    decrypted, x0 = decrypt_signal(
        samples=encrypted,
        meta=meta,
        x0=x0,
        r=r,
        chunk_size=block_samples
    )

    write_wav_mono(output_wav_path, decrypted, meta)
    return x0


def test_1_run(
    org,
    enc,
    dec,
    x0=0.1234567890123456,
    r=0.9876543210987654,
    errored_enc=None,
    org_format=None,
    new_format=None
):
    samples, meta = read_wav_mono(org)

    time_sample = 0.5
    block_samples = int(time_sample * meta["sample_rate"])

    encrypt_wav_file_realtime_like(org, enc, x0, r, block_samples)

    if errored_enc is not None:
        me.make_err(enc, errored_enc)
        enc = errored_enc


    if org_format is not None and new_format is not None:
        me.emulate_format_convertion(enc, org_format, new_format)
        # enc = enc+"."+str(new_format)

    decrypt_wav_file_realtime_like(enc, dec, x0, r, block_samples)