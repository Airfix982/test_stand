import argparse, json, wave
from pathlib import Path
import numpy as np

FRAME, CORE, GUARD = 320, 240, 80

def read_wav(path):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected mono 16-bit PCM WAV")
        p = w.getparams()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
    return p, x

def write_wav(path, p, x):
    with wave.open(str(path), "wb") as w:
        w.setparams(p)
        w.writeframes(np.asarray(np.clip(np.rint(x), -32768, 32767), dtype="<i2").tobytes())

def noise_like(seg, rng, gain=1.0):
    rms = max(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))), 300.0) * gain
    return np.clip(rng.normal(0.0, rms, len(seg)), -32768, 32767).astype(np.int16)

def linear_drift(x, ppm):
    n = int(round(len(x) * (1.0 + ppm * 1e-6)))
    pos = np.linspace(0.0, len(x) - 1.0, n)
    return np.interp(pos, np.arange(len(x)), x.astype(np.float64)).astype(np.int16)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/sync_v2_tx.wav")
    ap.add_argument("--out", default="/tmp/sync_stress")
    ap.add_argument("--seed", type=int, default=7777)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p, x = read_wav(args.src)
    rng = np.random.default_rng(args.seed)
    manifest = {}

    def save(name, y, events):
        path = out / f"{name}.wav"
        write_wav(path, p, y)
        manifest[name] = {"path": str(path), "samples": int(len(y)), "delta_samples": int(len(y) - len(x)), "events": events}
        print(f"{name:22s} samples={len(y)} delta={len(y)-len(x):+d}")

    # 1) Несколько timing-step по всему файлу.
    y, shift, ev = x.copy(), 0, []
    for frame, delta in [(900, -17), (1600, 36), (2300, -80), (3100, 55), (3900, -29), (4700, 44), (5500, -61)]:
        pos = frame * FRAME + shift
        if delta < 0:
            y = np.delete(y, np.s_[pos:pos - delta])
        else:
            y = np.concatenate((y[:pos], np.zeros(delta, dtype=np.int16), y[pos:]))
        shift += delta
        ev.append({"frame": frame, "type": "timing_step", "delta": delta, "physical_pos": pos})
    save("multi_steps", y, ev)

    # 2) Разные повреждения guard без изменения timeline.
    y, ev = x.copy(), []
    ops = [(700, "zero"), (1500, "noise"), (2300, "reverse"), (3100, "wrong_phase"), (3900, "attenuate"), (4700, "noise"), (5500, "zero")]
    for frame, kind in ops:
        a = frame * FRAME + CORE
        seg = y[a:a + GUARD].copy()
        if kind == "zero": y[a:a + GUARD] = 0
        elif kind == "noise": y[a:a + GUARD] = noise_like(seg, rng)
        elif kind == "reverse": y[a:a + GUARD] = seg[::-1]
        elif kind == "wrong_phase":
            b = (frame + 1) * FRAME + CORE
            y[a:a + GUARD] = x[b:b + GUARD]
        elif kind == "attenuate": y[a:a + GUARD] = (seg.astype(np.float64) * 0.12).astype(np.int16)
        ev.append({"frame": frame, "type": f"guard_{kind}"})
    save("guard_damage", y, ev)

    # 3) PLC-подобные 20 ms блоки: zero / repeat previous / noise.
    y, ev = x.copy(), []
    for frame, kind in [(1200, "zero"), (2200, "repeat_prev"), (3200, "noise"), (4200, "repeat_prev"), (5200, "zero")]:
        a = frame * FRAME
        seg = y[a:a + FRAME].copy()
        if kind == "zero": y[a:a + FRAME] = 0
        elif kind == "repeat_prev": y[a:a + FRAME] = y[a - FRAME:a]
        else: y[a:a + FRAME] = noise_like(seg, rng)
        ev.append({"frame": frame, "type": f"plc_{kind}"})
    save("plc_blocks", y, ev)

    # 4) Длинные плохие участки 0.5 / 1 / 2 s, длина timeline неизменна.
    y, ev = x.copy(), []
    bursts = [(1000, 8000, "zero"), (2800, 16000, "noise"), (4300, 32000, "repeat_prev")]
    for frame, n, kind in bursts:
        a = frame * FRAME
        seg = y[a:a + n].copy()
        if kind == "zero": y[a:a + n] = 0
        elif kind == "noise": y[a:a + n] = noise_like(seg, rng)
        else: y[a:a + n] = np.tile(y[a - FRAME:a], int(np.ceil(n / FRAME)))[:n]
        ev.append({"frame": frame, "type": f"burst_{kind}", "samples": n, "seconds": n / p.framerate})
    save("long_bad_pcm", y, ev)

    # 5) Постоянный clock drift. Это диагностический тест для будущего clock recovery.
    for ppm in (20, -20, 80):
        save(f"drift_{ppm:+d}ppm".replace("+", "p").replace("-", "m"), linear_drift(x, ppm), [{"type": "linear_clock_drift", "ppm": ppm}])

    # 6) Смешанный стресс: много разных повреждений, включая кластеры через несколько кадров.
    y, shift, ev = x.copy(), 0, []
    schedule = [
        (500, "guard_att", 0), (900, "guard_zero", 0), (908, "delete", 17), (917, "plc_repeat", 0), (929, "insert", 23),
        (1500, "guard_noise", 0), (1900, "plc_noise", 0), (2400, "guard_reverse", 0), (2410, "delete", 41), (2422, "plc_zero", 0), (2435, "insert", 37),
        (3000, "burst_noise", 3), (3600, "guard_wrong", 0), (4200, "plc_repeat", 0), (4207, "delete", 63), (4218, "guard_zero", 0), (4230, "insert", 52),
        (4700, "burst_repeat", 6), (5200, "guard_noise", 0), (5600, "delete", 28), (5612, "plc_noise", 0), (5630, "insert", 31)
    ]
    for frame, kind, value in schedule:
        pos = frame * FRAME + shift
        if kind == "delete":
            y = np.delete(y, np.s_[pos:pos + value]); shift -= value
        elif kind == "insert":
            y = np.concatenate((y[:pos], np.zeros(value, dtype=np.int16), y[pos:])); shift += value
        elif kind.startswith("guard_"):
            a = pos + CORE
            seg = y[a:a + GUARD].copy()
            if kind == "guard_zero": y[a:a + GUARD] = 0
            elif kind == "guard_noise": y[a:a + GUARD] = noise_like(seg, rng)
            elif kind == "guard_reverse": y[a:a + GUARD] = seg[::-1]
            elif kind == "guard_att": y[a:a + GUARD] = (seg.astype(np.float64) * 0.15).astype(np.int16)
            else:
                src = (frame + 1) * FRAME + CORE
                y[a:a + GUARD] = x[src:src + GUARD]
        elif kind.startswith("plc_"):
            a = pos
            seg = y[a:a + FRAME].copy()
            if kind == "plc_zero": y[a:a + FRAME] = 0
            elif kind == "plc_repeat": y[a:a + FRAME] = y[a - FRAME:a]
            else: y[a:a + FRAME] = noise_like(seg, rng)
        elif kind.startswith("burst_"):
            a, n = pos, value * FRAME
            seg = y[a:a + n].copy()
            if kind == "burst_noise": y[a:a + n] = noise_like(seg, rng)
            else: y[a:a + n] = np.tile(y[a - FRAME:a], value)
        ev.append({"frame": frame, "type": kind, "value": value, "shift_after": shift})
    save("mixed_realistic", y, ev)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()