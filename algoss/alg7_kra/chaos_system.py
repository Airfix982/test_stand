import hashlib
import numpy as np
import math

class ChaosSystem:
    def __init__(self, k=66., beta=1.44, warmup=4444, secret=4444):
        self.k = float(k)
        self.cos_beta = math.cos(beta)
        self.sin_beta = math.sin(beta)

        digest = hashlib.sha512(str(secret).encode()).digest()
        digits = "".join(str(b%10) for b in digest if b < 250)
        self.x = int(digits[:14]) / 10**14
        self.y = int(digits[14:28]) / 10**14

        self._byte = 0
        self._bits_remaining = 0

        for _ in range(warmup): self.step()

    def step(self):
        cocked_y = self.y + self.k * math.sin(self.x)
        next_x = self.cos_beta * self.x - self.sin_beta * cocked_y 
        next_y = self.sin_beta * self.x + self.cos_beta * cocked_y
        if not (math.isfinite(next_x) and math.isfinite(next_y)):
            raise ArithmeticError("Non-finite DKO state")
        self.x = next_x
        self.y = next_y

    def next_byte(self):
        self.step()
        u = self.y % 1.0
        return int(256.0*u)

    def next_bits(self, count):
        bits = np.empty(count, dtype=np.uint8)
        for i in range(count):
            if(self._bits_remaining == 0):
                self._byte = self.next_byte()
                self._bits_remaining = 8
            self._bits_remaining -=1
            bits[i] = (self._byte >> self._bits_remaining) & 1
        return bits