"""Bounded-memory DKO key cursor for realtime ALG7.

One ChaosSystem object, warmed in __init__ exactly once.
Keys are tied to absolute frame indexes. Skipped frames advance the same
trajectory. A bounded FIFO cache permits repeats and short backwards seeks.
Requests older than the window raise rather than secretly restart chaos.
"""

from collections import OrderedDict
import numpy as np

from algoss.alg7_kra.chaos_system import ChaosSystem


class ChaosKeyCursor:
    def __init__(self, seed=4444, bits_per_frame=174, warmup=4444,
                 k=66.0, beta=1.44, cache_frames=256,
                 max_skip_frames=1_000_000):
        self.bits_per_frame = int(bits_per_frame)
        self.cache_frames = int(cache_frames)
        self.max_skip_frames = int(max_skip_frames)
        if self.bits_per_frame <= 0:
            raise ValueError("bits_per_frame must be positive")
        if self.cache_frames < 1:
            raise ValueError("cache_frames must be positive")
        if self.max_skip_frames < 0:
            raise ValueError("max_skip_frames must be nonnegative")

        # Exactly one initialized and warmed oscillator per cursor lifetime.
        self._chaos = ChaosSystem(warmup=warmup, k=k, S=seed, beta=beta)
        self.next_index = 0
        self._cache = OrderedDict()  # frame index -> packed key bytes

    def get_key_for_frame(self, index):
        index = int(index)
        if index < 0:
            raise ValueError("Negative absolute frame index: %d" % index)

        packed = self._cache.get(index)
        if packed is not None:
            return self._unpack(packed)

        if index < self.next_index:
            oldest = next(iter(self._cache), self.next_index)
            raise RuntimeError(
                "DKO frame %d is older than cached window [%d, %d]. "
                "The oscillator was NOT restarted. Increase cache_frames "
                "or use state checkpoints for longer rewinds."
                % (index, oldest, self.next_index - 1)
            )

        gap = index - self.next_index
        if gap > self.max_skip_frames:
            raise RuntimeError(
                "DKO forward jump %d frames exceeds max_skip_frames=%d"
                % (gap, self.max_skip_frames)
            )

        # Advance through skipped absolute frame indexes as well.
        # Retain only the latest cache_frames keys, never the whole history.
        while self.next_index <= index:
            bits = self._chaos.next_bits(self.bits_per_frame)
            self._cache[self.next_index] = np.packbits(bits, bitorder="big").tobytes()
            self.next_index += 1
            if len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)

        return self._unpack(self._cache[index])

    def _unpack(self, packed):
        return np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8), bitorder="big"
        )[:self.bits_per_frame].copy()

    @property
    def cached_frames(self):
        return len(self._cache)

    @property
    def oldest_cached_index(self):
        return next(iter(self._cache), None)


if __name__ == "__main__":
    # Preserve bitstream even when bits_per_frame is not divisible by 8.
    bits_per_frame = 173
    cursor = ChaosKeyCursor(seed=4444, bits_per_frame=bits_per_frame)
    chaos_id = id(cursor._chaos)
    reference = ChaosSystem(warmup=1000, k=60.0, S=4444, beta=1.35)
    truth = [reference.next_bits(bits_per_frame) for _ in range(500)]

    for i in (0, 1, 2, 37, 37, 63, 65, 149, 80, 2, 149):
        assert np.array_equal(cursor.get_key_for_frame(i), truth[i]), i
    for i in range(150, 500):
        assert np.array_equal(cursor.get_key_for_frame(i), truth[i]), i
    assert cursor.cached_frames == 256
    assert id(cursor._chaos) == chaos_id
    assert cursor.oldest_cached_index == 500 - 256
    assert np.array_equal(cursor.get_key_for_frame(244), truth[244])

    try:
        cursor.get_key_for_frame(0)
        raise AssertionError("out-of-window request should have failed")
    except RuntimeError as exc:
        assert "NOT restarted" in str(exc)

    print("DKO bounded cursor: PASS (one warmup, no replay, <=256 cached frames)")
