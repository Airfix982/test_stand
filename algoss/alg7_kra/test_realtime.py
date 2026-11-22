#!/usr/bin/env python3
import math
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from pathlib import Path

# import script.utils as me
import numpy as np
import scipy.fftpack as scp
from scipy.io import wavfile
from scipy.signal import resample_poly
from scipy.signal import resample_poly, correlate

from .single_pitch import single_pitch
from .pseudo_speech import Pseudospeech_Synthetizer_class, parameters_pseudospeech
from .encryption import Encryption_class
from .speech_analyzer import Speech_Analyzer_class
from .DirectLPCSynthesizer import DirectLPCSynthesizer
from .chaos_system import ChaosSystem

BASE_DIR = Path(__file__).resolve().parent
SEED = 37
INPUT_WAV = BASE_DIR / "test11.wav"

my_analyzer = Speech_Analyzer_class(BASE_DIR / "speech_model.npz", BASE_DIR / "spherical_code.npz")
my_encryptor = Encryption_class(BASE_DIR / "spherical_code.npz")
my_ps = Pseudospeech_Synthetizer_class(BASE_DIR / "pseudospeech_model.npz", BASE_DIR / "spherical_code.npz")
KEY_BITS_PER_FRAME = int(my_encryptor.encryption_bits[2])

LPC_ORDER = 12
NB_BANDS = 9
FFT_SIZE = 320
FREQ_SIZE = FFT_SIZE // 2 + 1
COMPENSATION_8KHZ = np.array([
    0.800000011920929, 0.5, 0.4000000059604645, 0.28571000695228577,
    0.22222000360488892, 0.18182000517845154, 0.14285999536514282,
    0.11111000180244446, 0.20000000298023224
], dtype=np.float64)
EBAND_8KHZ = np.array([0, 2, 4, 7, 11, 16, 22, 30, 40], dtype=np.int32)


@contextmanager
def workdir(path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_mono_int16(path):
    fs, x = wavfile.read(path)
    if x.ndim == 2:
        x = np.mean(x.astype(np.float64), axis=1)
    if x.dtype != np.int16:
        if np.issubdtype(x.dtype, np.floating):
            x = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
        else:
            x = np.clip(x, -32768, 32767).astype(np.int16)
    return int(fs), x


def resample_audio(x, src_rate, dst_rate):
    if src_rate == dst_rate:
        return np.asarray(x, dtype=np.int16)
    g = math.gcd(int(src_rate), int(dst_rate))
    y = resample_poly(np.asarray(x, dtype=np.float64), dst_rate // g, src_rate // g)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def analyze_whole(samples_8k):
    # The native encoder uses fixed filenames from speech_analyzer.TEMP_DIR.
    # A unique temporary directory per call prevents collisions between
    # joblib/loky workers. It is removed immediately after analysis.
    from . import speech_analyzer as _speech_analyzer
    previous_dir = _speech_analyzer.TEMP_DIR
    with TemporaryDirectory(prefix="alg7_encoder_") as tmp_dir:
        _speech_analyzer.TEMP_DIR = Path(tmp_dir)
        try:
            with workdir(BASE_DIR):
                return my_analyzer.analyze_speech(samples_8k)
        finally:
            _speech_analyzer.TEMP_DIR = previous_dir


def pad_for_encoder(samples_8k):
    block = int(my_analyzer.speech_frame_size) * 4
    pad = (-len(samples_8k)) % block
    return (np.pad(samples_8k, (0, pad)) if pad else samples_8k), pad


def save_listening_wav(path, signal, fs, target_peak=0.90):
    y = np.asarray(signal, dtype=np.float64)
    peak = float(np.percentile(np.abs(y), 99.9)) if len(y) else 0.0
    if peak > 1e-12:
        y = y * (target_peak * 32767.0 / peak)
    wavfile.write(path, fs, np.clip(np.rint(y), -32768, 32767).astype(np.int16))


# ---------- exact 8 kHz LPC reconstruction ----------

def _interp_band_gain_8khz(e):
    e = np.asarray(e, dtype=np.float64)
    out = np.zeros(FREQ_SIZE, dtype=np.float64)
    for b in range(NB_BANDS - 1):
        start, end = int(EBAND_8KHZ[b] * 4), int(EBAND_8KHZ[b + 1] * 4)
        n = end - start
        for j in range(n):
            a = j / n
            out[start + j] = (1.0 - a) * e[b] + a * e[b + 1]
    out[-1] = 0.0
    return out


def _celt_lpc_12(ac):
    ac = np.asarray(ac, dtype=np.float64)
    lpc = np.zeros(LPC_ORDER, dtype=np.float64)
    err = float(ac[0])
    if err == 0.0:
        return lpc, err
    for i in range(LPC_ORDER):
        rr = ac[i + 1] + sum(lpc[j] * ac[i - j] for j in range(i))
        r = -rr / err
        lpc[i] = r
        for j in range((i + 1) // 2):
            a, b = lpc[j], lpc[i - 1 - j]
            lpc[j], lpc[i - 1 - j] = a + r * b, b + r * a
        err *= 1.0 - r * r
        if err < 0.001 * ac[0]:
            break
    return lpc, err


def _lpc_from_bands_8khz(band_energy):
    spectrum = _interp_band_gain_8khz(band_energy)
    ac = np.array((np.fft.irfft(spectrum, n=FFT_SIZE) * FFT_SIZE)[:LPC_ORDER + 1], dtype=np.float64)
    ac[0] += ac[0] * 1e-4 + 0.6842105263157895
    for i in range(1, LPC_ORDER + 1):
        ac[i] *= 1.0 - 6e-5 * i * i
    return _celt_lpc_12(ac)


def _lpc_from_cepstrum_8khz(cepstrum):
    c = np.asarray(cepstrum, dtype=np.float64).copy()
    c[0] += 4.0
    band_energy = np.power(10.0, scp.idct(c, type=2, norm="ortho")) * COMPENSATION_8KHZ
    return _lpc_from_bands_8khz(band_energy)


def params_to_full_features(pitch_values, energy_values, timbre_values):
    n = len(pitch_values)
    out = np.zeros((n, 23), dtype=np.float64)
    for i in range(n):
        D = np.asarray(timbre_values[:, i], dtype=np.float64).reshape(9)
        bands = np.maximum(float(energy_values[i]) * D * D, 1e-12)
        cep = scp.dct(np.log10(bands), type=2, norm="ortho")
        cep[0] -= 4.0
        out[i, :9] = cep
        out[i, 9] = (float(pitch_values[i]) - 100.0) / 50.0
        lpc, g = _lpc_from_cepstrum_8khz(cep)
        out[i, 10] = np.log10(max(float(g), 1e-30))
        out[i, 11:23] = lpc
    return out


# ---------- exact inverse of Speech_Analyzer_class.value_to_index ----------

def spher2cart_unit(phi):
    phi = np.asarray(phi, dtype=np.float64).reshape(8)
    x = np.empty(9, dtype=np.float64)
    prod = 1.0
    for i in range(7):
        x[i] = prod * np.cos(phi[i])
        prod *= np.sin(phi[i])
    x[7] = prod * np.cos(phi[7])
    x[8] = prod * np.sin(phi[7])
    return x


def indices_to_values(pitch_idx, energy_idx, timbre_idx):
    lo = float(my_analyzer.guard_low)
    rng = float(my_analyzer.indices_range)
    p = float(my_analyzer.pitch_period_min) + ((float(pitch_idx) - lo) / rng) * (
        float(my_analyzer.pitch_period_max) - float(my_analyzer.pitch_period_min)
    )
    log_e = float(my_analyzer.energy_min) + ((float(energy_idx) - lo) / rng) * (
        float(my_analyzer.energy_max) - float(my_analyzer.energy_min)
    )
    angles = np.asarray(timbre_idx, dtype=np.float64).reshape(8) * np.sqrt(8.0) / 2.0
    return p, 10.0 ** log_e, spher2cart_unit(angles)


# ---------- crypto wrappers; work with both original and locally patched encryption.py ----------
def encrypt_frame(p, e, t, key):
    b0, b1, b2 = map(int, my_encryptor.encryption_bits)

    p_enc = my_encryptor.Encrypt_translation(p, key[0:b0])

    e_enc = my_encryptor.Encrypt_translation(e, key[b0:b1])

    t_enc = my_encryptor.Encrypt_rotation(t, key[b1:b2])

    return p_enc, e_enc, t_enc


def decrypt_frame(p, e, t, key):
    b0, b1, b2 = map(int, my_encryptor.encryption_bits)

    p_dec = my_encryptor.Decrypt_translation(p, key[0:b0])

    e_dec = my_encryptor.Decrypt_translation(e, key[b0:b1])

    t_dec = my_encryptor.Decrypt_rotation(t, key[b1:b2])

    return p_dec, e_dec, t_dec

# ---------- pseudo-speech analysis; float64 avoids int16 square overflow ----------

def new_pseudo_state():
    return parameters_pseudospeech(my_ps.band_num, my_ps.harm_num_max)


def new_f0():
    bounds = np.array([my_ps.pitch_min, my_ps.pitch_max], dtype=np.float64) / my_ps.Fs
    return single_pitch(my_ps.flng, my_ps.harm_num_max, bounds)


def analyze_pseudo_frame(frame, f0, state):
    x = np.asarray(frame[my_ps.glng:], dtype=np.float64)
    X = np.fft.fft(x)
    rot = np.exp(-1j * 2.0 * np.pi * np.arange(my_ps.band_num) / my_ps.band_num)
    state.envelope_vector = 2.0 * np.real((X[:len(X) // 2] @ my_ps.filter_bank.T) * rot)
    state.energy_value = max(float(np.dot(x, x)), 1e-12)
    state.timbre_value = state.envelope_vector / max(np.linalg.norm(state.envelope_vector), 0.001)
    state.pitch_value = (my_ps.Fs / (2.0 * np.pi)) * f0.est(x, eps=2e-2)
    my_ps.value_to_index(state)
    return state.pitch_index, state.energy_index, state.timbre_index.copy()


# ---------- speech synthesis from physical values / indices ----------

def synthesize_values(pitch, energy, timbre, original_len, output_path, seed=12345):
    features = params_to_full_features(pitch, energy, timbre)
    synth = DirectLPCSynthesizer(fs=8000, frame_size=int(my_analyzer.speech_frame_size),
                                 periodic_mix=0.5, seed=seed)
    chunks = [synth.synthesize_frame(float(pitch[i]), float(features[i, 10]), features[i, 11:23])
              for i in range(len(pitch))]
    y = np.concatenate(chunks).astype(np.float64)[:original_len]
    print(f"{Path(output_path).name}: raw RMS={np.sqrt(np.mean(y*y)):.6f}, peak={np.max(np.abs(y)):.6f}")
    save_listening_wav(output_path, y, 8000)
    return y


def synthesize_indices(p_idx, e_idx, t_idx, original_len, output_path, seed=12345):
    n = len(p_idx)
    p = np.zeros(n); e = np.zeros(n); D = np.zeros((9, n))
    for i in range(n):
        p[i], e[i], D[:, i] = indices_to_values(p_idx[i], e_idx[i], t_idx[:, i])
    return synthesize_values(p, e, D, original_len, output_path, seed)


# ---------- diagnostics ----------

def modulo_diff(a, b, modulus):
    return (np.asarray(a) - np.asarray(b) + modulus / 2.0) % modulus - modulus / 2.0


def compare_indices(ref, got, title):
    pr, er, tr = map(lambda x: np.asarray(x, dtype=np.float64), ref)
    pg, eg, tg = map(lambda x: np.asarray(x, dtype=np.float64), got)
    n = min(len(pr), len(pg)); pr, er, tr, pg, eg, tg = pr[:n], er[:n], tr[:, :n], pg[:n], eg[:n], tg[:, :n]
    Q = float(my_encryptor.quantization_levels)
    pd, ed = modulo_diff(pg, pr, Q), modulo_diff(eg, er, Q)
    box = np.asarray(my_encryptor.boundary_box, dtype=np.float64).reshape(-1, 1)
    td = modulo_diff(tg, tr, box)
    tl2 = np.linalg.norm(td, axis=0)
    print(f"\n==== {title} ====")
    print(f"pitch: exact={np.count_nonzero(pd == 0)}/{n}, mean_abs={np.mean(np.abs(pd)):.6f}, max_abs={np.max(np.abs(pd)):.6f}")
    print(f"energy: exact={np.count_nonzero(ed == 0)}/{n}, mean_abs={np.mean(np.abs(ed)):.6f}, max_abs={np.max(np.abs(ed)):.6f}")
    print(f"timbre: mean_coord_abs={np.mean(np.abs(td)):.6f}, max_coord_abs={np.max(np.abs(td)):.6f}, mean_L2={np.mean(tl2):.6f}, max_L2={np.max(tl2):.6f}")


def test_pre_pseudospeech(input_path=INPUT_WAV, seed=SEED):
    fs, samples = load_mono_int16(input_path)
    samples_8k = resample_audio(samples, fs, 8000)
    original_len = len(samples_8k)
    padded, _ = pad_for_encoder(samples_8k)
    p_idx, e_idx, t_idx, p_val, e_val, D_val = analyze_whole(padded)
    n = len(p_idx)

    print(f"frames={n}")
    synthesize_values(p_val, e_val, D_val, original_len, BASE_DIR / "stage_0_physical.wav")
    synthesize_indices(p_idx, e_idx, t_idx, original_len, BASE_DIR / "stage_A_quantized.wav")

    p_err = []; e_ratio = []; d_err = []
    for i in range(n):
        p2, e2, d2 = indices_to_values(p_idx[i], e_idx[i], t_idx[:, i])
        p_err.append(p2 - p_val[i])
        e_ratio.append(e2 / max(float(e_val[i]), 1e-12))
        d_err.append(np.linalg.norm(d2 - D_val[:, i]))
    print("\nINDEX -> VALUE")
    print(f"pitch mean_abs={np.mean(np.abs(p_err)):.9f}, max_abs={np.max(np.abs(p_err)):.9f}")
    print(f"energy ratio median={np.median(e_ratio):.9f}, min={np.min(e_ratio):.9f}, max={np.max(e_ratio):.9f}")
    print(f"timbre mean_L2={np.mean(d_err):.9f}, max_L2={np.max(d_err):.9f}")

    rng = np.random.default_rng(seed)
    pd = np.zeros(n); ed = np.zeros(n); td = np.zeros((8, n))
    for i in range(n):
        key = rng.integers(0, 2, size=KEY_BITS_PER_FRAME, dtype=np.uint8)
        pe, ee, te = encrypt_frame(p_idx[i], e_idx[i], t_idx[:, i], key)
        pd[i], ed[i], td[:, i] = decrypt_frame(pe, ee, te, key)
    compare_indices((p_idx, e_idx, t_idx), (pd, ed, td), "CRYPTO ONLY")
    synthesize_indices(pd, ed, td, original_len, BASE_DIR / "stage_B_crypto_only.wav")


def encrypt_wav(input_path, output_path, seed=SEED):
    fs, samples = load_mono_int16(input_path)
    samples_8k = resample_audio(samples, fs, 8000)
    original_len = len(samples_8k)
    padded, _ = pad_for_encoder(samples_8k)
    p, e, t, _, _, _ = analyze_whole(padded)
    # rng, state, frames = np.random.default_rng(seed), new_pseudo_state(), []
    chaos = ChaosSystem()
    state, frames  = new_pseudo_state(), []
    for i in range(len(p)):
        # key = rng.integers(0, 2, size=KEY_BITS_PER_FRAME, dtype=np.uint8)
        key = chaos.next_bits(KEY_BITS_PER_FRAME)
        pe, ee, te = encrypt_frame(p[i], e[i], t[:, i], key)
        frames.append(my_ps.synthetize_frame(pe, ee, te, state))
    enc = np.concatenate(frames).astype(np.int16)
    wavfile.write(output_path, int(my_ps.Fs), enc)
    print(f"encrypted: frames={len(p)}, samples={len(enc)}, path={output_path}")
    return original_len, (p.copy(), e.copy(), t.copy())


def decrypt_wav(input_path, output_path, ref, original_len, seed=SEED):
    fs, enc = load_mono_int16(input_path)
    if fs != int(my_ps.Fs):
        raise ValueError(f"expected {my_ps.Fs} Hz, got {fs}")
    n = len(ref[0])
    frame_len = int(my_ps.flng + my_ps.glng)
    need = n * frame_len
    enc = np.pad(enc, (0, max(0, need - len(enc))))[:need]
    # rng, state, f0 = np.random.default_rng(seed), new_pseudo_state(), new_f0()
    chaos = ChaosSystem()
    state, f0 = new_pseudo_state(), new_f0()
    pd = np.zeros(n); ed = np.zeros(n); td = np.zeros((8, n))
    for i in range(n):
        frame = enc[i * frame_len:(i + 1) * frame_len]
        pr, er, tr = analyze_pseudo_frame(frame, f0, state)
        # key = rng.integers(0, 2, size=KEY_BITS_PER_FRAME, dtype=np.uint8)
        key = chaos.next_bits(KEY_BITS_PER_FRAME)
        pd[i], ed[i], td[:, i] = decrypt_frame(pr, er, tr, key)
    compare_indices(ref, (pd, ed, td), f"RAW AFTER DECRYPT {Path(input_path).name}")
    pd, ed, td = repair_decoded_indices(pd, ed, td)
    compare_indices(ref, (pd, ed, td), f"REPAIRED AFTER DECRYPT{Path(input_path).name}")
    synthesize_indices(pd, ed, td, original_len, output_path)
    return pd, ed, td

def opus_roundtrip_ffmpeg(input_wav, output_wav, bitrate="48k"):
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")

    input_wav = Path(input_wav)
    output_wav = Path(output_wav)

    # УНИКАЛЬНЫЙ temp для каждого файла / worker.
    opus_file = output_wav.with_suffix(".tmp.opus")

    try:
        subprocess.run([
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_wav),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libopus",
            "-application", "voip",
            "-b:a", bitrate,
            "-vbr", "off",
            str(opus_file)
        ], check=True)

        subprocess.run([
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(opus_file),
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(output_wav)
        ], check=True)

    finally:
        opus_file.unlink(missing_ok=True)

    print(f"Opus roundtrip saved: {output_wav}")

def run_clean():
    enc = BASE_DIR / "test11_enc.wav"
    dec = BASE_DIR / "test11_dec_clean.wav"
    original_len, ref = encrypt_wav(INPUT_WAV, enc)
    decrypt_wav(enc, dec, ref, original_len)

def run_opus():
    enc = BASE_DIR / "test11_enc.wav"
    enc_raw = BASE_DIR / "test11_enc_opus_raw.wav"
    enc_opus = BASE_DIR / "test11_enc_opus.wav"
    dec_opus = BASE_DIR / "test11_dec_opus.wav"

    original_len, ref = encrypt_wav(INPUT_WAV, enc)
    opus_roundtrip_ffmpeg(enc, enc_raw, "48k")
    align_codec_output(enc, enc_raw, enc_opus)
    decrypt_wav(enc_opus, dec_opus, ref, original_len)
# def run_opus():
#     enc = BASE_DIR / "test_sound6_enc.wav"
#     enc_opus = BASE_DIR / "test_sound6_enc_opus.wav"
#     dec_opus = BASE_DIR / "test_sound6_dec_opus.wav"
#     original_len, ref = encrypt_wav(INPUT_WAV, enc)
#     opus_roundtrip_ffmpeg(enc, enc_opus, "48k")
#     decrypt_wav(enc_opus, dec_opus, ref, original_len)

def align_codec_output(reference_wav, codec_wav, output_wav, max_lag=1600):
    fs1, ref = load_mono_int16(reference_wav)
    fs2, got = load_mono_int16(codec_wav)
    if fs1 != fs2:
        raise ValueError(f"fs mismatch: {fs1} vs {fs2}")

    a = ref.astype(np.float64)
    b = got.astype(np.float64)
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    corr = correlate(b0, a0, mode="full", method="fft")
    lags = np.arange(-(len(a0) - 1), len(b0))
    mask = np.abs(lags) <= max_lag
    lag = int(lags[mask][np.argmax(corr[mask])])

    if lag > 0:
        aligned = b[lag:]
    elif lag < 0:
        aligned = np.pad(b, (-lag, 0))
    else:
        aligned = b.copy()

    aligned = np.pad(aligned, (0, max(0, len(a) - len(aligned))))[:len(a)]

    gain = np.dot(a, aligned) / max(np.dot(aligned, aligned), 1e-12)
    aligned *= gain

    corrcoef = np.corrcoef(a, aligned)[0, 1]
    rmse = np.sqrt(np.mean((a - aligned) ** 2))

    print(f"codec alignment: lag={lag} samples ({1000*lag/fs1:.3f} ms), gain={gain:.6f}, corr={corrcoef:.6f}, rmse={rmse:.3f}")
    wavfile.write(output_wav, fs1, np.clip(np.rint(aligned), -32768, 32767).astype(np.int16))

def repair_decoded_indices(p, e, t):
    p = np.asarray(p, dtype=np.float64).copy()
    e = np.asarray(e, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64).copy()

    lo = float(my_analyzer.guard_low)
    hi = float(my_analyzer.guard_high)

    def repair_scalar(x):
        bad = (x < lo) | (x > hi)
        good = ~bad
        if np.any(good):
            x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), x[good])

        y = x.copy()
        for i in range(1, len(x) - 1):
            med = np.median(x[i-1:i+2])
            if abs(x[i] - med) > 2500:
                y[i] = med
        return np.clip(y, lo, hi)

    p = repair_scalar(p)
    e = repair_scalar(e)

    return p, e, t

def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "pre"
    if mode == "pre":
        test_pre_pseudospeech()
    elif mode == "clean":
        run_clean()
    elif mode == "opus":
        run_opus()
    elif mode == "all":
        test_pre_pseudospeech()
        run_clean()
        run_opus()
    else:
        raise SystemExit("usage: python -m ...test_realtime [pre|clean|opus|all]")

def test_7_run(
    org, enc, dec,
    errored_enc=None,
    org_format=None,
    new_format=None,
    codec=None,
    bitrate="48k",
    codec_bitrate=None,
    seed=4444,
):
    """Local ALG7: original WAV -> encrypted WAV -> optional codec -> decrypt.

    Saves AUDIO only, preserving every codec stage. Supports joblib/loky
    process parallelism with unique org/enc/dec paths.
    """
    import re

    if errored_enc is not None:
        raise ValueError("errored_enc is not used in ALG7 local pipeline")
    if org_format is not None and str(org_format).lower().lstrip(".") != "wav":
        raise ValueError("ALG7 expects WAV input")
    org, enc, dec = Path(org), Path(enc), Path(dec)
    if not org.is_file():
        raise FileNotFoundError(str(org))
    if len({org.resolve(), enc.resolve(), dec.resolve()}) != 3:
        raise ValueError("Input, encrypted and decrypted paths must differ")
    if enc.suffix.lower() != ".wav" or dec.suffix.lower() != ".wav":
        raise ValueError("enc and dec paths must end in .wav")
    enc.parent.mkdir(parents=True, exist_ok=True)
    dec.parent.mkdir(parents=True, exist_ok=True)

    fmt = str(new_format).lower().lstrip(".") if new_format is not None else "none"
    if fmt in ("none", "", "wav"):
        selected_codec = None
    elif fmt == "mp3":
        selected_codec = "mp3"
    elif fmt in ("ogg", "opus"):
        selected_codec = "opus"
    else:
        raise ValueError("new_format must be None, 'mp3', or 'ogg'")

    if selected_codec and codec is not None:
        named_codec = "opus" if str(codec).lower() in ("opus", "ogg") else str(codec).lower()
        if named_codec != selected_codec:
            raise ValueError("codec and new_format disagree: %r vs %r" % (codec, new_format))

    chosen_bitrate = str(codec_bitrate or bitrate or "48k").lower()
    if selected_codec and not re.fullmatch(r"[1-9][0-9]*k", chosen_bitrate):
        raise ValueError("bitrate must be like '64k', '96k', '320k'")

    # The original algorithm returns its in-memory reference indices.
    # These are used internally by decrypt_wav(), never written to disk.
    original_len, ref = encrypt_wav(str(org), str(enc), seed=int(seed))

    if selected_codec is None:
        decrypt_wav(str(enc), str(dec), ref, original_len, seed=int(seed))
        return

    original_enc = enc.with_name("org_" + enc.name)
    enc.replace(original_enc)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")

    extension = ".mp3" if selected_codec == "mp3" else ".ogg"
    compressed = Path(str(enc) + extension)  # enc1.wav.mp3 / enc1.wav.ogg
    raw = enc.with_name(enc.stem + "_" + selected_codec + "_raw.wav")
    aligned = enc
    # aligned = enc.with_name(enc.stem + "_" + selected_codec + "_aligned.wav")

    codec_args = (
        ["-ar", "44100", "-c:a", "libmp3lame", "-b:a", chosen_bitrate]
        if selected_codec == "mp3" else
        ["-ar", "48000", "-c:a", "libopus", "-application", "voip",
         "-vbr", "off", "-b:a", chosen_bitrate]
    )
    subprocess.run([
        ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(original_enc), "-map", "0:a:0", "-vn", "-ac", "1",
        *codec_args, str(compressed)
    ], check=True)
    subprocess.run([
        ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(compressed), "-map", "0:a:0", "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", str(raw)
    ], check=True)

    # The original local alignment uses clean enc.wav as reference.
    # The codec-processed WAV, not clean enc.wav, is sent to decrypt_wav().
    align_codec_output(str(original_enc), str(raw), str(aligned))
    raw.unlink(missing_ok=True)
    decrypt_wav(str(aligned), str(dec), ref, original_len, seed=int(seed))

if __name__ == "__main__":
    main()