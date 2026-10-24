import math
import hashlib

import numpy as np
from scipy.io import wavfile
import importlib



import script.utils as utils
importlib.reload(utils)
me = utils


def read_wav_mono_samples(path: str):
    fs, data = wavfile.read(path)

    if data.ndim != 1:
        raise ValueError("mono only")

    meta = {
        "sample_rate": int(fs),
        "dtype": str(data.dtype),
        "shape": list(data.shape),
    }
    return data, meta


def write_wav_samples(path: str, sample_rate: int, data: np.ndarray):
    wavfile.write(path, sample_rate, data)


def pcm_to_minus1_1(samples: np.ndarray) -> np.ndarray:
    if samples.dtype == np.uint8:
        x = samples.astype(np.float64)
        return (x / 255.0) * 2.0 - 1.0

    if samples.dtype == np.int16:
        return samples.astype(np.float64) / 32767.0

    if samples.dtype == np.int32:
        return samples.astype(np.float64) / 2147483647.0

    if np.issubdtype(samples.dtype, np.floating):
        x = samples.astype(np.float64)
        return np.clip(x, -1.0, 1.0)

    raise ValueError(f"unsupported dtype: {samples.dtype}")


def minus1_1_to_pcm(x: np.ndarray, dtype_name: str) -> np.ndarray:
    x = np.clip(x.astype(np.float64), -1.0, 1.0)

    if dtype_name == "uint8":
        y = np.rint(((x + 1.0) / 2.0) * 255.0)
        y = np.clip(y, 0, 255)
        return y.astype(np.uint8)

    if dtype_name == "int16":
        y = np.rint(x * 32768.0)
        y = np.clip(y, -32768, 32767)
        return y.astype(np.int16)

    if dtype_name == "int32":
        y = np.rint(x * 2147483648.0)
        y = np.clip(y, -(1 << 31), (1 << 31) - 1)
        return y.astype(np.int32)

    if dtype_name == "float32":
        return x.astype(np.float32)

    if dtype_name == "float64":
        return x.astype(np.float64)

    raise ValueError(f"unsupported target dtype: {dtype_name}")


def derive_session_params_from_secret(secret: str):
    """
    z0 взял как дробную часть
    omega > 2
    """
    h = hashlib.sha256(secret.encode("utf-8")).digest()

    a = int.from_bytes(h[0:8], "big") / float(2**64)
    b = int.from_bytes(h[8:16], "big") / float(2**64)
    c = int.from_bytes(h[16:20], "big") / float(2**64)

    eps = 1e-15

    z0 = ((10 ** 15) * (a + b + c)) % 1.0
    z0 = float(np.clip(z0, -1.0 + eps, 1.0 - eps))

    # omega > 2
    omega = 2.0 + abs((10 ** 15) * (a + b + c))

    warmup = 4444

    return z0, float(omega), int(warmup)


class ChebyshevSession:
    """
    Хаос стартует 1 раз и непрерывно идёт через все чанки.
    """

    def __init__(self, z0: float, omega: float, warmup: int = 0):
        self.z = float(z0)
        self.omega = float(omega)
        self.eps = 1e-15

        if warmup > 0:
            self.generate(warmup)

    def _step(self) -> float:
        z = min(max(self.z, -1.0 + self.eps), 1.0 - self.eps)
        z_next = math.cos(self.omega * math.acos(z))
        z_next = min(max(z_next, -1.0 + self.eps), 1.0 - self.eps)
        self.z = z_next
        return z_next

    def generate(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._step()
        return out


def preprocess_chunk_plain(x_chunk: np.ndarray):
    """
    Вход: x_chunk в [-1,1]
    Выход:
      x2i : uint8, целая часть [0..255]
      x2d : float64, дробная часть [0,1)
    """
    # x = np.clip(x_chunk.astype(np.float64), -1.0, 1.0)

    x2 = 256.0 * ((x_chunk + 1.0) / 2.0)

    x2_floor = np.floor(x2)
    x2i = np.clip(x2_floor, 0.0, 255.0).astype(np.uint8)

    x2d = x2 - x2_floor

    return x2i, x2d


def postprocess_chunk_cipher(xf: np.ndarray, x2d: np.ndarray) -> np.ndarray:
    """
    xc = (xf + x2d) / 256
    """
    xc = (xf.astype(np.float64) + x2d.astype(np.float64)) / 256.0
    return np.clip(xc, 0.0, np.nextafter(1.0, 0.0))


def preprocess_chunk_cipher(xc_chunk: np.ndarray):
    """
      xf  : uint8
      x2d : float64
    """
    xc = np.clip(xc_chunk.astype(np.float64), 0.0, np.nextafter(1.0, 0.0))
    u = xc * 256.0

    xf = np.floor(u).astype(np.uint8)
    x2d = u - np.floor(u)
    x2d = np.clip(x2d, 0.0, np.nextafter(1.0, 0.0))

    return xf, x2d


def postprocess_chunk_plain(x2i: np.ndarray, x2d: np.ndarray) -> np.ndarray:
    """
    x2 = x2i + x2d
    x = 2*(x2/256) - 1
    """
    x2 = x2i.astype(np.float64) + x2d.astype(np.float64)
    x = 2.0 * (x2 / 256.0) - 1.0
    return np.clip(x, -1.0, 1.0)


def build_scrambling_permutation(y: np.ndarray) -> np.ndarray:
    return np.argsort(y)[::-1].astype(np.int64)


def invert_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


def build_diffusion_sequence(y: np.ndarray) -> np.ndarray:
    """
    D = floor(|Y * 10^12| mod 256)
    """
    return np.floor(np.mod(np.abs(y) * 1e12, 256.0)).astype(np.uint8)


def encrypt_chunk(x_chunk: np.ndarray, y_chunk: np.ndarray) -> np.ndarray:
    if len(x_chunk) != len(y_chunk):
        raise ValueError("chunk and chaotic sequence length mismatch")

    x2i, x2d = preprocess_chunk_plain(x_chunk)

    perm = build_scrambling_permutation(y_chunk)
    x2i_scr = x2i[perm]

    d = build_diffusion_sequence(y_chunk)
    xf = np.bitwise_xor(x2i_scr, d).astype(np.uint8)

    xc = postprocess_chunk_cipher(xf, x2d)
    return xc


def decrypt_chunk(xc_chunk: np.ndarray, y_chunk: np.ndarray) -> np.ndarray:
    if len(xc_chunk) != len(y_chunk):
        raise ValueError("chunk and chaotic sequence length mismatch")

    xf, x2d = preprocess_chunk_cipher(xc_chunk)

    d = build_diffusion_sequence(y_chunk)
    x2i_scr = np.bitwise_xor(xf, d).astype(np.uint8)

    perm = build_scrambling_permutation(y_chunk)
    inv_perm = invert_permutation(perm)
    x2i = x2i_scr[inv_perm]

    x = postprocess_chunk_plain(x2i, x2d)
    return x


def split_into_chunks(samples: np.ndarray, chunk_size: int):
    for i in range(0, len(samples), chunk_size):
        yield samples[i:i + chunk_size]


def encrypt_wav_chebyshev_realtime(
    input_wav_path: str,
    encrypted_wav_path: str,
    secret: str,
    chunk_samples: int = 4096,
):
    src_samples, src_meta = read_wav_mono_samples(input_wav_path)
    fs = src_meta["sample_rate"]

    x = pcm_to_minus1_1(src_samples)

    z0, omega, warmup = derive_session_params_from_secret(secret)
    session = ChebyshevSession(z0=z0, omega=omega, warmup=warmup)

    enc_chunks = []

    for chunk in split_into_chunks(x, chunk_samples):
        y_chunk = session.generate(len(chunk))
        xc_chunk = encrypt_chunk(chunk, y_chunk)
        enc_chunks.append(xc_chunk)

    encrypted = np.concatenate(enc_chunks).astype(np.float32) if enc_chunks else np.array([], dtype=np.float32)

    write_wav_samples(encrypted_wav_path, fs, encrypted)


def decrypt_wav_chebyshev_realtime(
    encrypted_wav_path: str,
    decrypted_wav_path: str,
    secret: str,
    output_dtype: str = "int16",
    chunk_samples: int = 4096,
):
    enc_samples, enc_meta = read_wav_mono_samples(encrypted_wav_path)
    fs = enc_meta["sample_rate"]

    xc = enc_samples.astype(np.float64)
    xc = np.clip(xc, 0.0, np.nextafter(1.0, 0.0))

    z0, omega, warmup = derive_session_params_from_secret(secret)
    session = ChebyshevSession(z0=z0, omega=omega, warmup=warmup)

    dec_chunks = []

    for chunk in split_into_chunks(xc, chunk_samples):
        y_chunk = session.generate(len(chunk))
        x_chunk = decrypt_chunk(chunk, y_chunk)
        dec_chunks.append(x_chunk)

    x_rec = np.concatenate(dec_chunks).astype(np.float64) if dec_chunks else np.array([], dtype=np.float64)

    out_samples = minus1_1_to_pcm(x_rec, output_dtype)
    write_wav_samples(decrypted_wav_path, fs, out_samples)


def test_3_run(
    org,
    enc,
    dec,
    secret="nothing to see here",
    chunk_samples=4096,
    output_dtype="int16", org_format=None, new_format=None
):
    encrypt_wav_chebyshev_realtime(
        input_wav_path=org,
        encrypted_wav_path=enc,
        secret=secret,
        chunk_samples=chunk_samples,
    )
    

    if org_format is not None and new_format is not None:
        me.emulate_format_convertion(enc, org_format, new_format)
        # enc = enc+"."+str(new_format)
    # secret = "mothing to see here"
    decrypt_wav_chebyshev_realtime(
        encrypted_wav_path=enc,
        decrypted_wav_path=dec,
        secret=secret,
        output_dtype=output_dtype,
        chunk_samples=chunk_samples,
    )