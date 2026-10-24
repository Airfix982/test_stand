import time
import numpy as np
from scipy.io import wavfile
import pywt
import matplotlib.pyplot as plt
import script.utils as utils
import importlib



importlib.reload(utils)
me = utils

class ChaosKeys:

  def __init__(self, min_key_length, max_key_length, mod, secret1, secret2, secret3):

    # Decide on gamma and find etas
    self.gamma = 31 - 27 * secret1 / mod
    self.eta1 = 0.5 - np.sqrt(0.25 - np.floor(self.gamma / 4) / self.gamma)
    self.eta2 = 0.5 + np.sqrt(0.25 - np.floor(self.gamma / 4) / self.gamma)

    # Decide on x
    self.x = secret2 / mod
    self.keys = []

    # Decide on key list length
    if min_key_length != max_key_length:
      self.key_length = min_key_length + secret3 % (max_key_length - min_key_length)
    else:
      self.key_length = min_key_length

  # Use chaos maps to generate keys
  # Read into research article (at the top) for more info
  def generage_keys(self):
    bytestream = []
    bitcount = 8

    # Slightly modified to output bytes instead of bits
    while len(bytestream) < self.key_length:
      if self.eta1 <= self.x <= self.eta2:
        self.x = (self.gamma * self.x * (1 - self.x) % 1) / (self.gamma / 4 % 1)
      else:
        self.x = self.gamma * self.x * (1 - self.x) % 1
      if 0.1 <= self.x <= 0.6:
        bit = 1 if self.x * 10**10 % 1 > 0.5 else 0
        if bitcount < 8:
          bitcount += 1
          bytestream[-1] = bytestream[-1] * 2 + bit
        else:
          bitcount = 1
          bytestream.append(bit)

    self.keys = bytestream
    return self.keys

  def plot_keys(self):
    plt.plot(self.keys, "o")
    plt.xlabel("Keys Index")
    plt.ylabel("Key Value")
    plt.show()



RATE = 48000
BLOCKSIZE = 2400  # 2400 @ 48kHz = 50ms блок
ORG_LEN = None

def wrap_keys(xor_keys: np.ndarray, curr_key_idx: int, need_bytes: int) -> tuple[bytes, int]:
    keys = xor_keys[curr_key_idx:]
    while keys.size < need_bytes:
        keys = np.concatenate((keys, xor_keys), 0)

    curr_key_idx = (curr_key_idx + need_bytes) % xor_keys.size
    keys = keys[:need_bytes]
    return keys.tobytes(), curr_key_idx


def byte_xor(a: bytes, b: bytes) -> bytes:
    aa = np.frombuffer(a, dtype=np.uint8)
    bb = np.frombuffer(b, dtype=np.uint8)
    return np.bitwise_xor(aa, bb).tobytes()


def wav_to_int32_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = np.average(samples, axis=1)

    if samples.dtype == np.float32:
        samples = np.array(samples * 2147483647, dtype=np.int32)
    elif samples.dtype == np.int16:
        samples = samples.astype(np.int32) * 65538
    elif samples.dtype == np.uint8:
        samples = samples.astype(np.int32) * 16777216 - 2147483648
    else:
        samples = np.array(samples, dtype=np.int32)

    # в sounddevice callback у них форма (blocksize, 1)
    return samples.reshape((-1, 1))


def encrypt_wav_realtime_like(input_wav: str, encrypted_wav: str, xor_keys: np.ndarray,
                              emulate_realtime: bool = False) -> None:
    fs, samples = wavfile.read(input_wav)
    org_len = len(samples)
    if fs != RATE:
        print(f"[warn] input rate={fs}, orig rate={RATE}")

    data = wav_to_int32_mono(samples)

    curr_key_idx = 0
    out_blocks = []

    wavelet = pywt.Wavelet("db1")

    pos = 0
    while pos < data.shape[0]:
        t0 = time.perf_counter()

        take = min(BLOCKSIZE, data.shape[0] - pos)
        block = data[pos:pos + take].copy()

        audio_dwt = pywt.dwt(block, wavelet)[0]  
        audio_dwt = np.array(audio_dwt, dtype=np.int32)

        plain_bytes = np.ndarray.tobytes(audio_dwt)

        need_bytes = audio_dwt.size * 4
        key_bytes, curr_key_idx = wrap_keys(xor_keys, curr_key_idx, need_bytes)
        enc_bytes = byte_xor(plain_bytes, key_bytes)

        enc_i32 = np.frombuffer(enc_bytes, dtype=np.int32).reshape(audio_dwt.shape)
        out_blocks.append(enc_i32)

        pos += take

        if emulate_realtime:
            dt = time.perf_counter() - t0
            target = take / float(fs)
            if dt < target:
                time.sleep(target - dt)

    encrypted = np.vstack(out_blocks).astype(np.int32)
    wavfile.write(encrypted_wav, fs, encrypted)
    return org_len


def decrypt_wav_realtime_like(encrypted_wav: str, decrypted_wav: str, xor_keys: np.ndarray,
                              emulate_realtime: bool = False, original_len=None) -> None:
    fs, enc = wavfile.read(encrypted_wav)
    enc = wav_to_int32_mono(enc)

    curr_key_idx = 0
    out_blocks = []

    wavelet = pywt.Wavelet("db1")

    pos = 0
    while pos < enc.shape[0]:
        t0 = time.perf_counter()

        take = min(BLOCKSIZE, enc.shape[0] - pos)
        block = enc[pos:pos + take].copy()

        enc_bytes = np.ndarray.tobytes(block)

        need_bytes = block.size * 4
        key_bytes, curr_key_idx = wrap_keys(xor_keys, curr_key_idx, need_bytes)
        plain_bytes = byte_xor(enc_bytes, key_bytes)

        wavelet_data = np.frombuffer(plain_bytes, dtype=np.int32).reshape(block.shape)

        audio = pywt.idwt(wavelet_data, None, wavelet)
        audio = np.average(audio, axis=1)
        audio = audio.reshape((-1, 1)).astype(np.float32)

        out_blocks.append(audio)

        pos += take

        if emulate_realtime:
            dt = time.perf_counter() - t0
            target = take / float(fs)
            if dt < target:
                time.sleep(target - dt)

    decrypted = np.vstack(out_blocks).astype(np.float32)

    if original_len is not None:
        decrypted = decrypted[:original_len]

    m = np.max(np.abs(decrypted))
    if m > 0:
        decrypted_float_norm = decrypted / m
    else:
        decrypted_float_norm = decrypted

    decrypted_i16 = (decrypted_float_norm * 32767.0).astype(np.int16)
    wavfile.write(decrypted_wav, fs, decrypted_i16)




def test_2_run(org, enc, dec, secret1 = 444444444444444, secret2 = 555555555555555, secret3 = 666666666666666, errored_enc=None, org_format=None, new_format=None):
    # org = "test2/artefacts/input.wav"
    # enc = "test2/artefacts/encrypted.wav"
    # dec = "test2/artefacts/decrypted.wav"
    mod = 2**64 + 1
    min_len = 16384
    max_len = 32768

    xor_keys = np.array(
        ChaosKeys(min_len, max_len, mod, secret1, secret2, secret3).generage_keys(),
        dtype=np.uint8
    )
    emulate_real_time = True

    org_len = encrypt_wav_realtime_like(org, enc, xor_keys, emulate_realtime=emulate_real_time)

    if not errored_enc==None:
        me.make_err(enc, errored_enc)
        enc = errored_enc

    if org_format is not None and new_format is not None:
        me.emulate_format_convertion(enc, org_format, new_format)
        # enc = enc+"."+str(new_format)
    # secret1 = 444444444444444
    # secret2 = 555555555555555
    # secret3 = 666666666666667
    xor_keys2 = np.array(
        ChaosKeys(min_len, max_len, mod, secret1, secret2, secret3).generage_keys(),
        dtype=np.uint8
    )
    decrypt_wav_realtime_like(enc, dec, xor_keys2, emulate_realtime=emulate_real_time, original_len=org_len)
    print("done")
# 64 128 320
# 16 48 96