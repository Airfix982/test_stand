import wave
import numpy as np
import hashlib
import random
import math
from typing import Dict, List, Tuple
from scipy.interpolate import CubicSpline
import base64
import json
from scipy.signal import find_peaks
import algoss.script.utils as utils
import importlib



importlib.reload(utils)
me = utils
from PyEMD import EMD
from multiprocessing import Pool, cpu_count
import hashlib

def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

EMD_ = EMD()
EMD_.MAX_ITERATION = 1000


def read_wav_mono(path):
    with wave.open(path, "rb") as f:
        n_channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        n_frames = f.getnframes()
        comptype = f.getcomptype()
        raw = f.readframes(n_frames)
    
    if(comptype!="NONE"):
        raise ValueError("not compressed pcm wav only")
    if(n_channels != 1):
        raise ValueError("mono only")

    bits = 8 * sampwidth


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

    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(raw)


def clip_to_pcm_range(samples, meta):
    minimum = meta["min_value"]
    maximum = meta["max_value"]
    return np.clip(np.rint(samples), minimum, maximum).astype(np.int64)


def build_permutation(length, chunk_index, master_seed=0x123456789ABCDEF0):
    seed = generate_chunk_words(chunk_index, master_seed ^ 0xA5A5A5A5A5A5A5A5)[0]
    rng = random.Random(seed)
    idx = list(range(length))
    rng.shuffle(idx)
    return np.array(idx, dtype=np.int64)


def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(permutation), dtype=permutation.dtype)
    return inv


def apply_scrambling(samples, permutation):
    return samples[permutation]


def apply_inverse_scrambling(samples, permutation):
    inv = invert_permutation(permutation)
    return samples[inv]


# 2dclm + keystream


def build_keystream_u32(x, y):
    if len(x) != len(y):
        raise ValueError("x y must be the same length")

    scale = 10**16
    xi = np.floor(x * scale).astype(np.uint64)
    yi = np.floor(y * scale).astype(np.uint64)

    k = np.bitwise_xor(xi, yi) & np.uint64(0xFFFFFFFF)
    return k.astype(np.uint32)

def generate_2dclm(length, x0, y0, r1, r2, burn_in=1000):
    total = length + burn_in
    x = np.empty(total, dtype=np.float64)
    y = np.empty(total, dtype=np.float64)

    x[0] = x0
    y[0] = y0

    for i in range(total-1):
        xn = x[i]
        yn = y[i]
        x_next = r1 * math.cos(xn * (1.0 - xn * xn))
        y_next = r2 * math.cos(yn * (1.0 - xn))
        x[i+1] = abs(x_next) % 1.0
        y[i+1] = abs(y_next) % 1.0

    return x[burn_in:], y[burn_in:]


def build_keystream_words(x, y, bits_per_sample):
    if(len(x) != len(y)):
        raise ValueError("x y must be the same length")

    scale = 10**16
    xi = np.floor(x * scale).astype(np.uint64)
    yi = np.floor(y * scale).astype(np.uint64)
    mask = (1 << bits_per_sample) - 1

    k = ((xi ^ yi) & np.uint64(mask)).astype(np.uint64)
    return k








MASK64 = 0xFFFFFFFFFFFFFFFF

def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    z = z ^ (z >> 31)
    return z & MASK64


def generate_chunk_words(chunk_index: int, master_seed: int = 0x123456789ABCDEF0) -> list[int]:
    state = (chunk_index ^ master_seed) & MASK64
    words = []
    for _ in range(8):
        state = splitmix64(state)
        words.append(state)
    return words


def derive_2dclm_params_from_chunk_index(chunk_index: int, master_seed: int = 0x123456789ABCDEF0):
    words = generate_chunk_words(chunk_index, master_seed)
    denom = float(2**64)

    x0 = ((words[0] ^ words[1]) / denom)
    y0 = ((words[2] ^ words[3]) / denom)
    r1 = ((words[4] ^ words[5]) / denom) + 2.9
    r2 = ((words[6] ^ words[7]) / denom) + 2.9

    eps = 1e-15
    x0 = min(max(x0, eps), 1.0 - eps)
    y0 = min(max(y0, eps), 1.0 - eps)
    return x0, y0, r1, r2


# matrix n padding

def pad_to_multiple(samples, multiple):
    pad_len = (-len(samples)) % multiple
    if(pad_len==0):
        return samples.copy(), 0

    padded = np.pad(samples, (0, pad_len), mode="reflect")
    return padded, pad_len


def reshape_to_matrix(samples: np.ndarray, row_len):
    if(len(samples) % row_len != 0):
        raise ValueError("Длина должна делиться на row_len")
    rows = len(samples) // row_len
    return samples.reshape(rows, row_len)


def flatten_matrix(mat):
    return mat.reshape(-1)

# EMD


def match_rms(reference, signal, eps=1e-12):
    reference = np.asarray(reference, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    ref_rms = np.sqrt(np.mean(reference ** 2))
    sig_rms = np.sqrt(np.mean(signal ** 2))

    gain = ref_rms / max(sig_rms, eps)
    return signal * gain

def encrypt_residue_xor_mantissa(residue, ks_row, mantissa_bits_to_xor=12):
    residue32 = np.asarray(residue, dtype=np.float32)
    residue_u32 = residue32.view(np.uint32)

    sign_exp = residue_u32 & np.uint32(0xFF800000)
    mantissa = residue_u32 & np.uint32(0x007FFFFF)

    xor_mask = np.uint32((1 << mantissa_bits_to_xor) - 1)
    ks_masked = ks_row.astype(np.uint32) & xor_mask

    mantissa_enc = mantissa ^ ks_masked
    out_u32 = sign_exp | mantissa_enc

    out_f32 = out_u32.view(np.float32)
    return out_f32.astype(np.float64)

def decrypt_residue_xor_mantissa(residue_enc, ks_row, mantissa_bits_to_xor=12):
    residue32 = np.asarray(residue_enc, dtype=np.float32)
    residue_u32 = residue32.view(np.uint32)

    sign_exp = residue_u32 & np.uint32(0xFF800000)
    mantissa = residue_u32 & np.uint32(0x007FFFFF)

    if mantissa_bits_to_xor <= 0 or mantissa_bits_to_xor > 23:
        raise ValueError("mantissa_bits_to_xor must be in range 1..23")

    xor_mask = np.uint32((1 << mantissa_bits_to_xor) - 1)
    ks_masked = ks_row.astype(np.uint32) & xor_mask

    mantissa_dec = mantissa ^ ks_masked
    out_u32 = sign_exp | mantissa_dec

    out_f32 = out_u32.view(np.float32)
    return out_f32.astype(np.float64)

def emd_decompose(signal, max_imfs=10):
    signal = np.asarray(signal, dtype=np.float64)

    emd = EMD_
    imfs_res = emd.emd(signal, max_imf=max_imfs)
    if imfs_res is None or len(imfs_res) == 0:
        return [], signal.copy()
    imfs_res = np.asarray(imfs_res, dtype=np.float64)


    imfs = imfs_res[:-1]
    residue = imfs_res[-1].copy()

    return [imf.copy() for imf in imfs], residue


def reconstruct_from_imfs_and_residue(imfs, residue):
    if not imfs:
        return residue.copy()
    acc = np.sum(np.vstack(imfs), axis=0)
    return acc + residue


# Residue Encryption

def signed_to_unsigned_words(values, bits):
    mask = (1 << bits) - 1
    return (values.astype(np.int64) & mask).astype(np.uint64)

def unsigned_to_signed_words(values, bits):
    sign_bit = 1 << (bits - 1)
    full = 1 << bits
    v = values.astype(np.int64)
    return np.where(v >= sign_bit, v - full, v).astype(np.int64)

def float_residue_to_pcm_int(residue, meta):
    return clip_to_pcm_range(np.rint(residue), meta)


def encrypt_residue(residue, ks_row, meta, mode="xor", mantissa_bits_to_xor=12):
    bits = meta["bits_per_sample"]

    if mode == "xor":
        return encrypt_residue_xor_mantissa(
            residue=residue,
            ks_row=ks_row,
            mantissa_bits_to_xor=mantissa_bits_to_xor
        )

    if mode == "modadd":
        r_int = float_residue_to_pcm_int(residue, meta)
        mod = 1 << bits
        r_u = signed_to_unsigned_words(r_int, bits)
        out_u = (r_u + ks_row.astype(np.uint64)) % mod
        out_s = unsigned_to_signed_words(out_u, bits)
        return out_s.astype(np.float64)

    raise ValueError(f"Unknown mode: {mode}")


def decrypt_residue(residue_enc, ks_row, meta, mode="xor", mantissa_bits_to_xor=12):
    bits = meta["bits_per_sample"]

    if mode == "xor":
        return decrypt_residue_xor_mantissa(
            residue_enc=residue_enc,
            ks_row=ks_row,
            mantissa_bits_to_xor=mantissa_bits_to_xor
        )

    if mode == "modadd":
        r_int = float_residue_to_pcm_int(residue_enc, meta)
        mod = 1 << bits
        r_u = signed_to_unsigned_words(r_int, bits)
        out_u = (r_u - ks_row.astype(np.uint64)) % mod
        out_s = unsigned_to_signed_words(out_u, bits)
        return out_s.astype(np.float64)

    raise ValueError("unknown mod")

#per-row / per-chunk crypto
def encrypt_matrix_rows(mat, ks_mat, meta, itr, mode, emd_max_imfs, emd_max_siftings, mantissa_bits_to_xor=12):
    out = mat.astype(np.float64).copy()

    for _ in range(itr):
        new_mat = np.empty_like(out)
        for i in range(out.shape[0]):
            sig = out[i, :]
            imfs, residue = emd_decompose(sig, max_imfs=emd_max_imfs)
            residue_enc = encrypt_residue(
                            residue,
                            ks_mat[i, :],
                            meta,
                            mode=mode,
                            mantissa_bits_to_xor=mantissa_bits_to_xor
                        )
            sig_enc = reconstruct_from_imfs_and_residue(imfs, residue_enc)
            new_mat[i, :] = sig_enc
        out = new_mat
    return out


def decrypt_matrix_rows(mat, ks_mat, meta, itr, mode, emd_max_imfs, emd_max_siftings):
    out = mat.astype(np.float64).copy()

    for _ in range(itr):
        new_mat = np.empty_like(out)
        for i in range(out.shape[0]):
            sig = out[i, :]
            imfs, residue_enc = emd_decompose(sig, max_imfs=emd_max_imfs)
            residue = decrypt_residue(
                        residue_enc,
                        ks_mat[i, :],
                        meta,
                        mode=mode,
                        mantissa_bits_to_xor=20
                    )
            sig_dec = reconstruct_from_imfs_and_residue(imfs, residue)
            new_mat[i, :] = sig_dec
        out = new_mat
    return out


def encrypt_chunk(chunk_samples, chunk_index, master_seed, meta, row_len, itr, mode, emd_max_imfs, x0, y0, r1, r2):
    # x0, y0, r1, r2 = derive_2dclm_params_from_chunk_index(chunk_index, master_seed)
    mantissa_bits_to_xor = 20
    perm = build_permutation(len(chunk_samples), chunk_index, master_seed)
    scrambled = apply_scrambling(chunk_samples, perm)

    padded, pad_len = pad_to_multiple(scrambled, row_len)
    mat = reshape_to_matrix(padded.astype(np.float64), row_len)

    # x, y = generate_2dclm(len(padded), x0, y0, r1, r2)
    x, y = generate_2dclm(len(padded), x0, y0, r1, r2, burn_in=0 if chunk_index==0 else 1)

    if mode == "xor":
        ks = build_keystream_u32(x, y)
    else:
        ks = build_keystream_words(x, y, meta["bits_per_sample"])

    ks_mat = reshape_to_matrix(ks, row_len)

    enc_mat = encrypt_matrix_rows(
        mat=mat,
        ks_mat=ks_mat,
        meta=meta,
        itr=itr,
        mode=mode,
        emd_max_imfs=emd_max_imfs,
        emd_max_siftings=0,
        mantissa_bits_to_xor=mantissa_bits_to_xor
    )

    enc_flat = flatten_matrix(enc_mat)
    if pad_len:
        enc_flat = enc_flat[:-pad_len]

    enc_int = clip_to_pcm_range(enc_flat, meta)
    return enc_int



def decrypt_chunk(enc_chunk_samples, chunk_index, master_seed, meta, row_len, itr, mode, emd_max_imfs, x0, y0, r1, r2):
    # x0, y0, r1, r2 = derive_2dclm_params_from_chunk_index(chunk_index, master_seed)

    perm = build_permutation(len(enc_chunk_samples), chunk_index, master_seed)

    padded, pad_len = pad_to_multiple(enc_chunk_samples, row_len)
    mat = reshape_to_matrix(padded.astype(np.float64), row_len)

    # x, y = generate_2dclm(len(padded), x0, y0, r1, r2)
    x, y = generate_2dclm(len(padded), x0, y0, r1, r2, burn_in=0 if chunk_index==0 else 1)

    if mode == "xor":
        ks = build_keystream_u32(x, y)
    else:
        ks = build_keystream_words(x, y, meta["bits_per_sample"])

    ks_mat = reshape_to_matrix(ks, row_len)

    dec_mat = decrypt_matrix_rows(
        mat=mat,
        ks_mat=ks_mat,
        meta=meta,
        itr=itr,
        mode=mode,
        emd_max_imfs=emd_max_imfs,
        emd_max_siftings=0
    )

    dec_flat = flatten_matrix(dec_mat)
    if pad_len:
        dec_flat = dec_flat[:-pad_len]

    dec_int = clip_to_pcm_range(dec_flat, meta)

    restored = apply_inverse_scrambling(dec_int, perm)
    return clip_to_pcm_range(restored, meta)







def split_into_chunks(samples, chunk_size):
    return [samples[i: i + chunk_size] for i in range(0, len(samples), chunk_size)]

def encrypt_signal(samples, meta, chunk_size, row_len, itr, mode, emd_max_imfs, master_seed):

    x0, y0, r1, r2 = derive_2dclm_params_from_chunk_index(0, master_seed)
    x, y = generate_2dclm(1, x0, y0, r1, r2, burn_in=1000)
    x0, y0 = float(x[0]), float(y[0])

    chunks = split_into_chunks(samples, chunk_size)
    enc_chunks = []

    for chunk_index, chunk in enumerate(chunks):
        enc_chunk = encrypt_chunk(
            chunk_samples=chunk,
            chunk_index=chunk_index,
            master_seed=master_seed,
            meta=meta,
            row_len=row_len,
            itr=itr,
            mode=mode,
            emd_max_imfs=emd_max_imfs,
            x0=x0,
            y0=y0,
            r1=r1,
            r2=r2
        )
        enc_chunks.append(enc_chunk)
        length = len(chunk) + (-len(chunk)) % row_len
        xx, yy = generate_2dclm(length,x0, y0, r1, r2,burn_in=0 if chunk_index == 0 else 1)
        x0, y0 = float(xx[-1]), float(yy[-1])

    encrypted = np.concatenate(enc_chunks).astype(np.int64) if enc_chunks else np.array([], dtype=np.int64)
    return encrypted

def decrypt_signal(samples, meta, chunk_size, row_len, itr, mode, emd_max_imfs, master_seed):

    x0, y0, r1, r2 = derive_2dclm_params_from_chunk_index(0, master_seed)
    x, y = generate_2dclm(1, x0, y0, r1, r2, burn_in=1000)
    x0, y0 = float(x[0]), float(y[0])
    
    chunks = split_into_chunks(samples, chunk_size)
    dec_chunks = []

    for chunk_index, enc_chunk in enumerate(chunks):
        dec_chunk = decrypt_chunk(
            enc_chunk_samples=enc_chunk,
            chunk_index=chunk_index,
            master_seed=master_seed,
            meta=meta,
            row_len=row_len,
            itr=itr,
            mode=mode,
            emd_max_imfs=emd_max_imfs,
            x0=x0,
            y0=y0,
            r1=r1,
            r2=r2
        )
        dec_chunks.append(dec_chunk)
        length = len(enc_chunk) + (-len(enc_chunk)) % row_len
        xx, yy = generate_2dclm(length,x0, y0, r1, r2,burn_in=0 if chunk_index == 0 else 1)
        x0, y0 = float(xx[-1]), float(yy[-1])

    decrypted = np.concatenate(dec_chunks).astype(np.int64) if dec_chunks else np.array([], dtype=np.int64)
    return decrypted



def test_5_run(org, enc, dec, errored_enc=None, org_format=None,
               new_format=None, codec=None, bitrate=None):
    # ONLY WAV/codec transport is modified. EMD cipher functions remain unchanged.
    import subprocess
    from pathlib import Path

    chunk_size = 2048
    row_len = 2048
    itr = 1
    mode = "xor"
    emd_max_imfs = 3
    master_seed = 0x123456789ABCDEF0

    samples, meta = read_wav_mono(org)
    Path(enc).parent.mkdir(parents=True, exist_ok=True)
    Path(dec).parent.mkdir(parents=True, exist_ok=True)
    Path(dec).unlink(missing_ok=True)  # Do not mistake an older dec.wav for a successful run.

    encrypted = encrypt_signal(
        samples=samples, meta=meta, chunk_size=chunk_size,
        row_len=row_len, itr=itr, mode=mode,
        emd_max_imfs=emd_max_imfs, master_seed=master_seed,
    )
    write_wav_mono(enc, encrypted, meta)
    # No use of original samples after this line; no match_rms(samples, decrypted).
    del samples

    if errored_enc is not None:
        me.make_err(enc, errored_enc)
        enc = errored_enc

    input_for_decrypt = str(enc)
    if new_format is not None:
        fmt = str(new_format).lower().lstrip('.')
        if fmt not in ('mp3', 'ogg', 'opus'):
            raise ValueError('new_format must be None, mp3, ogg or opus')
        use_opus = fmt in ('ogg', 'opus')
        if codec is not None and str(codec).lower() not in (
            ('opus', 'libopus', 'ogg') if use_opus else ('mp3', 'libmp3lame')
        ):
            raise ValueError('codec and new_format disagree')
        br = str(bitrate or ('96k' if use_opus else '128k'))
        codec_file = str(enc) + ('.ogg' if use_opus else '.mp3')
        decoded_file = str(Path(enc).with_name(Path(enc).stem +
                          ('_opus_decoded.wav' if use_opus else '_mp3_decoded.wav')))
        encode_opts = (['-ar', '48000', '-c:a', 'libopus', '-application', 'audio']
                       if use_opus else ['-ar', '44100', '-c:a', 'libmp3lame'])
        subprocess.run([
            'ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-i', str(enc), '-map', '0:a:0', '-vn', '-ac', '1',
            *encode_opts, '-b:a', br, codec_file,
        ], check=True)
        pcm_codec = {1: 'pcm_u8', 2: 'pcm_s16le', 4: 'pcm_s32le'}.get(meta['sample_width_bytes'])
        if pcm_codec is None:
            raise ValueError('unsupported source PCM width')
        subprocess.run([
            'ffmpeg', '-y', '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-i', codec_file, '-map', '0:a:0', '-vn', '-ac', '1',
            '-ar', str(meta['sample_rate']), '-c:a', pcm_codec, decoded_file,
        ], check=True)
        input_for_decrypt = decoded_file
        print('ALG5 codec:', enc, '->', codec_file, '->', decoded_file, flush=True)

    enc_samples, enc_meta = read_wav_mono(input_for_decrypt)
    if (enc_meta['sample_rate'] != meta['sample_rate'] or
            enc_meta['sample_width_bytes'] != meta['sample_width_bytes']):
        raise ValueError('codec did not restore original PCM rate/width')
    expected_len = len(encrypted)
    actual_len = len(enc_samples)
    delta = actual_len - expected_len

    if delta != 0:
        max_delta = max(
            32, int(round(meta["sample_rate"] * 0.005))
        )

        if new_format is None or abs(delta) > max_delta:
            raise ValueError(
                f"ALG5: unexpected sample count: "
                f"{expected_len} -> {actual_len}"
            )

        if delta > 0:
            enc_samples = enc_samples[:expected_len]
        else:
            silence = 128 if meta["sample_width_bytes"] == 1 else 0
            enc_samples = np.pad(
                enc_samples,
                (0, -delta),
                mode="constant",
                constant_values=silence
            )

        print(
            f"ALG5: codec length aligned {delta:+d} samples",
            flush=True
        )

    if new_format is not None:
        changed = int(np.count_nonzero(enc_samples != encrypted))
        print('ALG5 encrypted samples changed by codec: %d/%d' %
              (changed, len(encrypted)), flush=True)
        if not changed:
            raise RuntimeError('codec round-trip had no effect on ciphertext')
    del encrypted
    if new_format is not None:
        Path(enc).replace(
            Path(enc).with_name("org_" + Path(enc).name)
        )
        write_wav_mono(enc, enc_samples, enc_meta)
        Path(decoded_file).unlink(missing_ok=True)

    decrypted = decrypt_signal(
        samples=enc_samples, meta=enc_meta, chunk_size=chunk_size,
        row_len=row_len, itr=itr, mode=mode,
        emd_max_imfs=emd_max_imfs, master_seed=master_seed,
    )
    # Never correct the result using source PCM; output is solely decrypted data.
    decrypted = clip_to_pcm_range(decrypted, enc_meta)
    write_wav_mono(dec, decrypted, enc_meta)


    # diff = samples.astype(np.int64) - decrypted.astype(np.int64)


