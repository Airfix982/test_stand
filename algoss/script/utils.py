import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import os
import hashlib
from pathlib import Path

def load_wav_bytes(path: str) -> tuple[int, np.ndarray]:
    fs, data = wavfile.read(path)
    if data.dtype.kind == "f":
        m = np.max(np.abs(data))
        if m > 0:
            data = data / m
        data = (data * 32767.0).astype(np.int16)
    elif data.dtype == np.int32:
        data = (data >> 16).astype(np.int16)
    elif data.dtype == np.uint8:
        data = ((data.astype(np.int16) - 128) << 8).astype(np.int16)
    else:
        data = data.astype(np.int16)
    raw = np.frombuffer(data.tobytes(), dtype=np.uint8)
    return fs, raw

def save_wav_bytes(bytes_: tuple[int, np.ndarray], path: str):
    fs, raw = bytes_
    samples = np.frombuffer(raw.tobytes(), dtype=np.int16)
    wavfile.write(path, fs, samples)


def make_err(org, err):

    fs, bytes_1 = load_wav_bytes(org)
    bytes_2 = bytes_1.copy()
    bytes_2[0] ^= 0b00010000
    save_wav_bytes((fs, bytes_2), err)

def emulate_format_convertion(path, org_format, new_format):
    # 64 128 320 mp3

    # 16 48 96 opus
    bitrate="96k"
    codec="libopus"
    # print("codec " + codec + " bitrate " + bitrate)
    print("convert from " + org_format + " to " + new_format)
    new_path = path+"."+str(new_format)
    AudioSegment.from_file(path, org_format).export(new_path, new_format, bitrate=bitrate, codec=codec)
    AudioSegment.from_file(new_path, new_format).export(path, org_format)


def _md5(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        return f"<md5 error: {e}>"

# def emulate_format_convertion(path, org_format, new_format):
#     p = Path(path)
#     new_path = str(p) + f".{new_format}"
#     audio_in = AudioSegment.from_file(path, format=org_format)
#     bitrate="64k"
#     codec="libopus"
#     audio_in.export(new_path, format=new_format, bitrate=bitrate)#codec="libopus"
#     print(bitrate)
#     audio_tmp = AudioSegment.from_file(new_path, format=new_format)
#     audio_tmp.export(path, format=org_format)
