"""Control experiment: unchanged audio payload through the existing SuperSpinner/VDO service runtime.

No encryption is applied. Never use this module to protect private speech.
"""
import numpy as np
from experiment.ALG2.ALG2_syncCipherServiceRuntime import main, N

class PassthroughCipher:
    def __init__(self, secret):
        pass

    def encode(self, voice, index):
        data = np.asarray(voice, dtype=np.float32).reshape(N)
        return np.clip(data, -1.0, 1.0)

    def decode(self, wire, index):
        data = np.asarray(wire, dtype=np.float32).reshape(N)
        return np.clip(data, -1.0, 1.0)

    def close(self):
        pass

if __name__ == '__main__':
    main(PassthroughCipher, 'control_no_encryption', 'call')
