python3 - "/home/cossie/nears/test_stand/algoss/alg5_emd/test_encrypted/enc1.wav" "/home/cossie/nears/test_stand/algoss/alg5_emd/test_encrypted/enc1_opus_decoded.wav" <<'PY'
import sys
from pathlib import Path
import numpy as np
from algoss.alg5_emd import emd_v3 as alg

tx_path = Path(sys.argv[1])
rx_path = Path(sys.argv[2])

# Только зашифрованные данные. Исходный WAV не открываем.
tx, tx_meta = alg.read_wav_mono(str(tx_path))
rx, rx_meta = alg.read_wav_mono(str(rx_path))

if (tx_meta["sample_rate"] != rx_meta["sample_rate"] or
        tx_meta["sample_width_bytes"] != rx_meta["sample_width_bytes"]):
    raise ValueError("Sample rate / PCM width mismatch")

# Компенсация возможного расхождения на один отсчёт.
delta = len(rx) - len(tx)

if delta == 1:
    rx = rx[:len(tx)]
elif delta == -1:
    rx = np.pad(rx, (0, 1), mode="constant")
elif delta != 0:
    raise ValueError(f"Unexpected length mismatch: {delta}")

print("TX length:", len(tx))
print("RX length:", len(rx))
print("Codec changed samples:", np.count_nonzero(tx != rx))

CORRECT_KEY = 0x123456789ABCDEF0
WRONG_KEY = CORRECT_KEY ^ 1

params = dict(
    chunk_size=2048,
    row_len=2048,
    itr=1,
    mode="xor",
    emd_max_imfs=3,
)

def decrypt_and_save(data, meta, key, name):
    print(f"Decrypting {name}, key={key:#x}", flush=True)

    result = alg.decrypt_signal(
        samples=data,
        meta=meta,
        master_seed=key,
        **params
    )

    path = rx_path.parent / name
    alg.write_wav_mono(str(path), result, meta)

    print("Saved:", path, flush=True)
    return result

# 1. Без кодека, правильный ключ.
clean = decrypt_and_save(
    tx, tx_meta, CORRECT_KEY, "dec_no_codec.wav"
)

# 2. После Opus, правильный ключ.
correct = decrypt_and_save(
    rx, rx_meta, CORRECT_KEY, "dec_opus_correct.wav"
)

# 3. После Opus, неправильный ключ.
wrong = decrypt_and_save(
    rx, rx_meta, WRONG_KEY, "dec_opus_wrong.wav"
)

print(
    "Correct/wrong differing samples:",
    np.count_nonzero(correct != wrong),
    "/", len(correct)
)

print("EXPERIMENT COMPLETE")
PY