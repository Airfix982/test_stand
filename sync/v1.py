from dataclasses import dataclass
import time

import numpy as np
from scipy.signal import find_peaks, firwin


_GUARD_SEEDS = [
    10227, 10139, 10918, 10636,
    10820, 10345, 10578, 10889
]


@dataclass
class SyncV1Config:
    rate: int = 16000
    frameSamples: int = 320
    coreSamples: int = 240
    guardSamples: int = 80

    spinnerSize: int = 8

    acquireThreshold: float = 0.28
    trackThreshold: float = 0.20

    acquireConfirm: int = 8

    # ±120 samples = ±7.5 ms.
    trackRadius: int = 120

    # При проверке train каждый следующий guard
    # может гулять только совсем немного.
    trainRadius: int = 8

    # Если период между соседними guard изменился
    # сильнее чем на 24 samples = 1.5 ms,
    # текущий payload считаем повреждённым.
    maxTimingJump: int = 24

    periodAlpha: float = 0.02

    # Guards постоянно подтягивают wall-clock predictor.
    wallAlpha: float = 0.08

    # Сколько последних кадров держать для SEARCH/REACQUIRE.
    searchWindowFrames: int = 16

    # При spinnerSize=8 разрешаем predictor ошибаться
    # меньше чем на половину цикла.
    maxPredictionErrorFrames: float = 3.5
    codeMargin: float = 0.04


def makeGuardBank(rate=16000, samples=80, count=8):
    if count < 2 or count > len(_GUARD_SEEDS):
        raise ValueError(f"spinnerSize must be 2..{len(_GUARD_SEEDS)}")

    taps = firwin(31, [300, 3400], pass_zero=False, fs=rate)
    guards = []

    for seed in _GUARD_SEEDS[:count]:
        rng = np.random.default_rng(seed)
        raw = rng.choice((-1.0, 1.0), size=samples + 30)

        x = np.convolve(raw, taps, mode="same")[15:15 + samples]

        ramp = min(6, samples // 4)
        if ramp:
            r = np.sin(np.linspace(0, np.pi / 2, ramp)) ** 2
            w = np.ones(samples)
            w[:ramp] = r
            w[-ramp:] = r[::-1]
            x *= w

        x /= np.sqrt(np.mean(x * x) + 1e-12)
        guards.append(x.astype(np.float32))

    return np.stack(guards)


# Оставляем для совместимости со старым кодом.
def makeGuard(rate=16000, samples=80):
    return makeGuardBank(rate, samples, 2)[0]


class _SampleBuffer:
    def __init__(self):
        self.base = 0
        self.data = np.empty(0, dtype=np.float32)

    @property
    def end(self):
        return self.base + len(self.data)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        if len(x):
            self.data = np.concatenate((self.data, x))

    def slice(self, start, end):
        if start < self.base or end > self.end or end < start:
            return None

        return self.data[
            start - self.base:
            end - self.base
        ]

    def discardBefore(self, index):
        index = min(max(index, self.base), self.end)
        n = index - self.base

        if n:
            self.data = self.data[n:]
            self.base = index


class FixedGuardSyncV1:
    def __init__(self, cfg=None, onFrame=None, onEvent=None):
        self.cfg = cfg or SyncV1Config()

        if self.cfg.rate != 16000:
            raise ValueError("SyncV1 currently expects 16000 Hz")

        if self.cfg.coreSamples + self.cfg.guardSamples != self.cfg.frameSamples:
            raise ValueError("coreSamples + guardSamples must equal frameSamples")

        self.guards = makeGuardBank(
            self.cfg.rate,
            self.cfg.guardSamples,
            self.cfg.spinnerSize
        )

        self.guardNorms = np.linalg.norm(self.guards, axis=1)

        self.onFrame = onFrame or (lambda *args: None)
        self.onEvent = onEvent or (lambda *args: None)

        self.buf = _SampleBuffer()

        self.state = "SEARCH"

        self.lastGuard = None
        self.lastFrame = -1

        # Оценка physical RX samples / logical frame.
        self.periodEstimate = float(self.cfg.frameSamples)

        # wall time, при котором логический frame 0
        # должен был иметь свой guard.
        self.epochWall = None

        self.tailIndex = 0
        self.tailWall = None

        self.lastReacquireSearchEnd = -1

        self.stats = {
            "frames_ok": 0,
            "frames_lost": 0,
            "acquisitions": 0,
            "reacquisitions": 0,
            "track_misses": 0,
            "spinner_mismatches": 0
        }

    def push(self, samples, tailWall=None):
        samples = np.asarray(samples, dtype=np.float32)

        if not len(samples):
            return

        self.buf.append(samples)

        self.tailIndex = self.buf.end
        self.tailWall = (
            time.monotonic()
            if tailWall is None
            else tailWall
        )

        while True:
            if self.state == "SEARCH":
                progressed = self._acquire()
            elif self.state == "LOCKED":
                progressed = self._track()
            else:
                progressed = self._reacquire()

            if not progressed:
                break

        self._prune()

    def finish(self):
        # Нужен главным образом offline-тестам:
        # позволяет обработать последний полный guard,
        # хотя справа уже нет trackRadius samples.
        while self.state == "LOCKED":
            if not self._track(final=True):
                break

    def _wallAt(self, sampleIndex):
        if self.tailWall is None:
            return None

        return self.tailWall - (
            self.tailIndex - sampleIndex
        ) / self.cfg.rate

    def _predictFrame(self, wall):
        if wall is None or self.epochWall is None:
            return None

        frameSec = self.cfg.frameSamples / self.cfg.rate
        return (wall - self.epochWall) / frameSec

    def _alignPredictor(self, frameIndex, guardPos, hard=False):
        wall = self._wallAt(guardPos)

        if wall is None:
            return

        frameSec = self.cfg.frameSamples / self.cfg.rate
        observedEpoch = wall - frameIndex * frameSec

        if self.epochWall is None or hard:
            self.epochWall = observedEpoch
        else:
            a = self.cfg.wallAlpha
            self.epochWall = (
                (1.0 - a) * self.epochWall
                + a * observedEpoch
            )

    def _bankCorrRange(self, lo, hi):
        L = self.cfg.guardSamples

        lo = max(int(lo), self.buf.base)
        hi = min(int(hi), self.buf.end - L)

        if hi < lo:
            return None, None

        x = self.buf.slice(lo, hi + L)

        if x is None or len(x) < L:
            return None, None

        energy = np.sqrt(
            np.convolve(
                x * x,
                np.ones(L, dtype=np.float32),
                mode="valid"
            )
        )

        corr = []

        for i, guard in enumerate(self.guards):
            dots = np.correlate(x, guard, mode="valid")

            corr.append(
                dots / (
                    energy * self.guardNorms[i]
                    + 1e-12
                )
            )

        positions = np.arange(lo, lo + len(corr[0]))
        return positions, np.stack(corr)

    def _bestAnyGuard(self, lo, hi):
        positions, corr = self._bankCorrRange(lo, hi)

        if positions is None:
            return None, None, 0.0

        flat = int(np.argmax(corr))
        code, posIndex = np.unravel_index(
            flat,
            corr.shape
        )

        return (
            int(positions[posIndex]),
            int(code),
            float(corr[code, posIndex])
        )

    def _bestExpectedGuard(self, code, lo, hi):
        positions, corr = self._bankCorrRange(lo, hi)

        if positions is None:
            return None, 0.0, 0.0

        row = corr[code]
        i = int(np.argmax(row))

        value = float(row[i])

        others = np.delete(corr[:, i], code)
        second = float(np.max(others)) if len(others) else -1.0
        margin = value - second

        return int(positions[i]), value, margin

    def _findTrains(self, lo, hi, threshold):
        positions, corr = self._bankCorrRange(lo, hi)

        if positions is None:
            return []

        bestCorr = np.max(corr, axis=0)
        bestCode = np.argmax(corr, axis=0)

        peaks, _ = find_peaks(
            bestCorr,
            height=threshold,
            distance=max(20, self.cfg.guardSamples // 2)
        )

        trains = []

        for peak in peaks:
            firstPos = int(positions[peak])
            firstCode = int(bestCode[peak])
            firstValue = float(bestCorr[peak])

            column = corr[:, peak]
            others = np.delete(column, firstCode)
            firstMargin = firstValue - float(np.max(others))

            if firstMargin < self.cfg.codeMargin:
                continue

            values = [firstValue]
            margins = [firstMargin]
            positionsFound = [firstPos]

            ok = True

            for j in range(1, self.cfg.acquireConfirm):
                expectedPos = firstPos + j * self.cfg.frameSamples
                expectedCode = (firstCode + j) % self.cfg.spinnerSize

                pos, value, margin = self._bestExpectedGuard(
                    expectedCode,
                    expectedPos - self.cfg.trainRadius,
                    expectedPos + self.cfg.trainRadius
                )

                if pos is None or value < threshold or margin < self.cfg.codeMargin:
                    ok = False
                    break

                positionsFound.append(pos)
                values.append(value)
                margins.append(margin)

            if ok:
                trains.append({
                    "pos": firstPos,
                    "phase": firstCode,
                    "score": float(np.mean(values)),
                    "min_margin": float(np.min(margins)),
                    "positions": positionsFound
                })

        return trains

    def _resolveFrameIndex(self, phase, guardPos):
        wall = self._wallAt(guardPos)
        predicted = self._predictFrame(wall)

        if predicted is None:
            return phase, float(phase)

        cycle = self.cfg.spinnerSize

        k0 = int(round(
            (predicted - phase) / cycle
        ))

        candidates = []

        for dk in range(-3, 4):
            n = phase + (k0 + dk) * cycle

            if n > self.lastFrame:
                candidates.append(n)

        if not candidates:
            return None, predicted

        best = min(
            candidates,
            key=lambda n: abs(n - predicted)
        )

        return int(best), float(predicted)

    def _coreAt(self, guardPos):
        return self.buf.slice(
            guardPos - self.cfg.coreSamples,
            guardPos
        )

    def _emit(self, frameIndex, core, status, **meta):
        out = np.zeros(
            self.cfg.frameSamples,
            dtype=np.float32
        )

        if status == "ok" and core is not None:
            out[:self.cfg.coreSamples] = core

        if status == "ok":
            self.stats["frames_ok"] += 1
        else:
            self.stats["frames_lost"] += 1

        meta["spinner_phase"] = (
            frameIndex % self.cfg.spinnerSize
        )

        self.onFrame(
            frameIndex,
            out,
            status,
            meta
        )

    def _event(self, name, **fields):
        fields["event"] = name
        fields["state"] = self.state
        self.onEvent(fields)

    def _acquire(self):
        confirmSpan = (
            self.cfg.guardSamples
            + (self.cfg.acquireConfirm - 1)
            * (
                self.cfg.frameSamples
                + self.cfg.trainRadius
            )
        )

        if len(self.buf.data) < confirmSpan:
            return False

        window = (
            self.cfg.searchWindowFrames
            * self.cfg.frameSamples
        )

        lo = max(
            self.buf.base,
            self.buf.end - window
        )

        hi = self.buf.end - confirmSpan

        trains = self._findTrains(
            lo,
            hi,
            self.cfg.acquireThreshold
        )

        if not trains:
            return False

        # На старте берём самый ранний подтверждённый train.
        train = min(
            trains,
            key=lambda t: t["pos"]
        )

        guardPos = train["pos"]
        phase = train["phase"]

        # В V1 RX запускается ДО TX.
        # Поэтому первый найденный spinner cycle
        # соответствует первым 0..7 frames сессии.
        frameIndex = phase

        for n in range(frameIndex):
            self._emit(
                n,
                None,
                "lost",
                reason="before_initial_lock"
            )

        core = self._coreAt(guardPos)

        if core is None:
            return False

        self._emit(
            frameIndex,
            core.copy(),
            "ok",
            guard_pos=guardPos,
            corr=train["score"],
            timing_error_samples=0.0,
            observed_period="",
            period_estimate=self.periodEstimate,
            predicted_frame=frameIndex
        )

        self.lastGuard = guardPos
        self.lastFrame = frameIndex

        self._alignPredictor(
            frameIndex,
            guardPos,
            hard=True
        )

        self.state = "LOCKED"
        self.stats["acquisitions"] += 1

        self._event(
            "acquired",
            guard_pos=guardPos,
            frame=frameIndex,
            spinner_phase=phase,
            corr=train["score"]
        )

        return True

    def _track(self, final=False):
        frameIndex = self.lastFrame + 1

        expectedPos = (
            self.lastGuard
            + self.periodEstimate
        )

        if final:
            if self.buf.end < expectedPos + self.cfg.guardSamples:
                return False

            lo = max(
                self.buf.base,
                expectedPos - self.cfg.trackRadius
            )

            hi = min(
                expectedPos + self.cfg.trackRadius,
                self.buf.end - self.cfg.guardSamples
            )

            if hi < lo:
                return False

        else:
            needEnd = (
                expectedPos
                + self.cfg.trackRadius
                + self.cfg.guardSamples
            )

            if self.buf.end < needEnd:
                return False

            lo = expectedPos - self.cfg.trackRadius
            hi = expectedPos + self.cfg.trackRadius

        guardPos, phase, corr = self._bestAnyGuard(
            lo,
            hi
        )

        expectedPhase = (
            frameIndex % self.cfg.spinnerSize
        )

        if (
            guardPos is None
            or corr < self.cfg.trackThreshold
        ):
            self.state = "REACQUIRE"
            self.stats["track_misses"] += 1

            self._event(
                "track_miss",
                expected_frame=frameIndex,
                expected_guard=expectedPos,
                best_corr=corr
            )

            return True

        if phase != expectedPhase:
            self.state = "REACQUIRE"

            self.stats["track_misses"] += 1
            self.stats["spinner_mismatches"] += 1

            self._event(
                "spinner_mismatch",
                expected_frame=frameIndex,
                expected_phase=expectedPhase,
                observed_phase=phase,
                guard_pos=guardPos,
                corr=corr
            )

            return True

        observedPeriod = (
            guardPos - self.lastGuard
        )

        timingError = (
            guardPos - expectedPos
        )

        wall = self._wallAt(guardPos)
        predictedFrame = self._predictFrame(wall)

        core = self._coreAt(guardPos)

        if core is None:
            return False

        # Сильный скачок произошёл внутри интервала
        # между двумя guard. Номер frame известен,
        # но payload этого frame не считаем безопасным.
        damaged = (
            abs(
                observedPeriod
                - self.cfg.frameSamples
            )
            > self.cfg.maxTimingJump
        )

        self._emit(
            frameIndex,
            None if damaged else core.copy(),
            "lost" if damaged else "ok",
            guard_pos=guardPos,
            corr=corr,
            timing_error_samples=float(timingError),
            observed_period=float(observedPeriod),
            period_estimate=float(self.periodEstimate),
            predicted_frame=predictedFrame,
            reason=(
                "timing_jump"
                if damaged
                else ""
            )
        )

        if (
            abs(
                observedPeriod
                - self.cfg.frameSamples
            )
            <= 4
        ):
            a = self.cfg.periodAlpha

            self.periodEstimate = (
                (1.0 - a) * self.periodEstimate
                + a * observedPeriod
            )

        self.lastGuard = guardPos
        self.lastFrame = frameIndex

        # Вот здесь guard постоянно "подтягивает"
        # time predictor обратно к реальной сетке.
        self._alignPredictor(
            frameIndex,
            guardPos,
            hard=False
        )

        return True

    def _reacquire(self):
        # Не запускаем тяжёлый search после каждого
        # крошечного PCM chunk.
        if (
            self.buf.end
            - self.lastReacquireSearchEnd
            < self.cfg.frameSamples
        ):
            return False

        self.lastReacquireSearchEnd = self.buf.end

        confirmSpan = (
            self.cfg.guardSamples
            + (self.cfg.acquireConfirm - 1)
            * (
                self.cfg.frameSamples
                + self.cfg.trainRadius
            )
        )

        if len(self.buf.data) < confirmSpan:
            return False

        window = (
            self.cfg.searchWindowFrames
            * self.cfg.frameSamples
        )

        lo = max(
            self.buf.base,
            self.buf.end - window
        )

        hi = self.buf.end - confirmSpan

        trains = self._findTrains(
            lo,
            hi,
            self.cfg.acquireThreshold
        )

        if not trains:
            return False

        candidates = []

        for train in trains:
            frameIndex, predicted = (
                self._resolveFrameIndex(
                    train["phase"],
                    train["pos"]
                )
            )

            if frameIndex is None:
                continue

            predictionError = abs(
                frameIndex - predicted
            )

            if (
                predictionError
                > self.cfg.maxPredictionErrorFrames
            ):
                continue

            candidates.append({
                "train": train,
                "frame": frameIndex,
                "predicted": predicted,
                "prediction_error": predictionError
            })

        if not candidates:
            return False

        # Сначала доверяем wall-clock prediction,
        # потом силе PN train.
        best = min(
            candidates,
            key=lambda c: (
                c["prediction_error"],
                -c["train"]["score"],
                c["train"]["pos"]
            )
        )

        train = best["train"]
        frameIndex = best["frame"]
        guardPos = train["pos"]

        oldFrame = self.lastFrame
        oldGuard = self.lastGuard

        # Всё между последним подтверждённым frame
        # и новым найденным считаем потерянным.
        for n in range(
            oldFrame + 1,
            frameIndex
        ):
            self._emit(
                n,
                None,
                "lost",
                reason="reacquire_gap"
            )

        core = self._coreAt(guardPos)

        physicalGap = (
            guardPos - oldGuard
        )

        logicalGap = (
            frameIndex - oldFrame
        )

        # Если reacquire говорит, что это просто
        # следующий frame, но физический интервал
        # внезапно сильно отличился от 320 samples,
        # именно этот payload считаем повреждённым.
        recoveredDamaged = (
            logicalGap == 1
            and abs(
                physicalGap
                - self.cfg.frameSamples
            )
            > self.cfg.maxTimingJump
        )

        if core is None:
            recoveredDamaged = True

        self._emit(
            frameIndex,
            (
                None
                if recoveredDamaged
                else core.copy()
            ),
            (
                "lost"
                if recoveredDamaged
                else "ok"
            ),
            guard_pos=guardPos,
            corr=train["score"],
            timing_error_samples="",
            observed_period="",
            period_estimate=float(self.periodEstimate),
            predicted_frame=best["predicted"],
            prediction_error_frames=best["prediction_error"],
            reason=(
                "reacquire_timing_jump"
                if recoveredDamaged
                else ""
            )
        )

        # После discontinuity старый sample-rate estimate
        # пока не тащим за собой.
        self.periodEstimate = float(
            self.cfg.frameSamples
        )

        self.lastGuard = guardPos
        self.lastFrame = frameIndex

        # После подтверждённого spinner train
        # заново точно привязываем wall predictor.
        self._alignPredictor(
            frameIndex,
            guardPos,
            hard=True
        )

        self.state = "LOCKED"
        self.stats["reacquisitions"] += 1

        self._event(
            "reacquired",
            guard_pos=guardPos,
            recovered_frame=frameIndex,
            spinner_phase=train["phase"],
            predicted_frame=best["predicted"],
            prediction_error_frames=best["prediction_error"],
            skipped_frames=max(
                0,
                frameIndex - oldFrame - 1
            ),
            recovered_status=(
                "lost"
                if recoveredDamaged
                else "ok"
            )
        )

        return True

    def _prune(self):
        keep = (
            self.cfg.searchWindowFrames
            * self.cfg.frameSamples
            + self.cfg.coreSamples
            + self.cfg.trackRadius
        )

        self.buf.discardBefore(
            max(
                self.buf.base,
                self.buf.end - keep
            )
        )