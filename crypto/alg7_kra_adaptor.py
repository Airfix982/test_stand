"""alg7_kra_adaptor — compatibility wrapper for original ALG7/KRA speech cipher.

Uses ORIGINAL Encryption_class methods and ORIGINAL numpy default_rng(seed)
per-frame key generation. Does NOT change encryption.py or any ALG7 files.
Transport only supplies indices and PCM.
"""
import importlib
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import threading
import numpy as np
from scipy.signal import resample_poly
from scipy.fftpack import dct, idct
from .frame_port import FrameCryptoPlugin, FrameContext, FRAME, check_batch, check_frame

_COMP = np.array([.800000011920929,.5,.4000000059604645,.28571000695228577,.22222000360488892,.18182000517845154,.14285999536514282,.11111000180244446,.2])
_EBANDS = np.array([0,2,4,7,11,16,22,30,40], dtype=int)

# The original Speech_Analyzer_class invokes './speech_encoder' and reads/writes
# 'temp/...' relative to the current directory. Keep the legacy implementation
# unchanged; confine its relative file accesses to --algo-dir.
_LEGACY_ANALYZER_LOCK = threading.RLock()

@contextmanager
def _analyzer_workdir(directory):
    with _LEGACY_ANALYZER_LOCK:
        prev = Path.cwd()
        (directory / 'temp').mkdir(parents=True, exist_ok=True)
        os.chdir(str(directory))
        try:
            yield
        finally:
            os.chdir(str(prev))


def _load_module_tree(directory):
    directory = Path(directory).expanduser().resolve()
    package_name = '_nears_user_speech_algo'
    needed = ('speech_analyzer.py','pseudo_speech.py','encryption.py','single_pitch.py','DirectLPCSynthesizer.py',
              'speech_model.npz','spherical_code.npz','pseudospeech_model.npz','speech_encoder')
    missing = [name for name in needed if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError('Missing in --algo-dir %s: %s' % (directory, ', '.join(missing)))
    pitch_lib = directory / 'cpp/lib/single_pitch.so'
    if not pitch_lib.is_file():
        raise FileNotFoundError('Missing native estimator: %s' % pitch_lib)
    if not os.access(directory / 'speech_encoder', os.X_OK):
        raise PermissionError('speech_encoder must be executable: chmod +x %s' % (directory / 'speech_encoder'))
    # The supplied pseudo_speech.py uses `from single_pitch import single_pitch`
    # (absolute import). Expose its original directory to the Python import system.
    existing = sys.modules.get('single_pitch')
    if existing is not None:
        previous = getattr(existing, '__file__', None)
        if not previous or Path(previous).resolve() != (directory / 'single_pitch.py'):
            raise RuntimeError('Conflicting imported single_pitch module: %s' % previous)
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    if package_name in sys.modules:
        loaded = Path(sys.modules[package_name].__path__[0]).resolve()
        if loaded != directory:
            raise RuntimeError('Cannot load different algo-dir within one Python process')
    else:
        spec = importlib.util.spec_from_file_location(package_name, directory / '__init__.py', submodule_search_locations=[str(directory)]) if (directory / '__init__.py').exists() else None
        if spec:
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
        else:
            import types
            module = types.ModuleType(package_name)
            module.__path__ = [str(directory)]
            sys.modules[package_name] = module
    loaded = {}
    for mod in ('speech_analyzer','pseudo_speech','encryption','single_pitch','DirectLPCSynthesizer'):
        loaded[mod] = importlib.import_module('.' + mod, package_name)
    return directory, loaded


def _sphere_cart(phi):
    phi = np.asarray(phi, dtype=np.float64).reshape(8)
    out = np.empty(9, dtype=np.float64)
    prod = 1.0
    for i in range(7):
        out[i] = prod * np.cos(phi[i])
        prod *= np.sin(phi[i])
    out[7], out[8] = prod*np.cos(phi[7]), prod*np.sin(phi[7])
    return out


def _lpc_cep(cep):
    c = np.asarray(cep, dtype=np.float64).copy()
    c[0] += 4.0
    e = np.power(10.0, idct(c, type=2, norm='ortho'))*_COMP
    spectrum = np.zeros(161, dtype=np.float64)
    for b in range(8):
        start, end = int(_EBANDS[b]*4), int(_EBANDS[b+1]*4)
        spectrum[start:end] = e[b] + (e[b+1] - e[b])*np.arange(end-start)/(end-start)
    ac = (np.fft.irfft(spectrum, n=320)*320)[:13]
    ac[0] += ac[0]*1e-4 + 0.6842105263157895
    for i in range(1,13):
        ac[i] *= 1 - 6e-5*i*i
    lpc = np.zeros(12, dtype=np.float64)
    err = float(ac[0])
    if err == 0.0:
        return lpc, err
    for i in range(12):
        rr = ac[i+1] + sum(lpc[j]*ac[i-j] for j in range(i))
        r = -rr/err
        lpc[i] = r
        for j in range((i+1)//2):
            a,b = lpc[j],lpc[i-1-j]
            lpc[j],lpc[i-1-j] = a + r*b, b + r*a
        err *= 1-r*r
        if err < .001*ac[0]:
            break
    return lpc, err

class OriginalNumpyRngCursor:
    """Exact stream of original: rng.integers(0,2,size=B,dtype=np.uint8).

    * Frame 0 draws first B bits; frame n draws after n earlier draws.
    * Missing frames are consumed, never reseeded to a new seed.
    * On a backwards index, resets from original seed and replays.
    * Caches the current frame to make repeated get_key_for_frame(n) idempotent.
    This class provides synchronization of key *position*, not cryptographic
    security. The user's original seed and RNG remain unchanged.
    """

    def __init__(self, seed, bits_per_frame, max_skip_frames=1_000_000):
        self.seed = int(seed)
        self.bits_per_frame = int(bits_per_frame)
        self.max_skip_frames = int(max_skip_frames)
        if self.bits_per_frame <= 0:
            raise ValueError('ALG7 key length must be positive')
        self._reset()

    def _reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.next_index = 0
        self._cached_index = None
        self._cached_bits = None

    def get_key_for_frame(self, index):
        index = int(index)
        if index < 0:
            raise ValueError('Negative absolute frame index: %d' % index)
        if index == self._cached_index:
            return self._cached_bits.copy()
        if index < self.next_index:
            # Original PCG64 is replayed; no alternative random-access KDF.
            self._reset()
        count = index - self.next_index
        if count > self.max_skip_frames:
            raise RuntimeError('ALG7 RNG jump %d frames exceeds limit; frame index/session epoch may be wrong' % count)
        for _ in range(count):
            self.rng.integers(0, 2, size=self.bits_per_frame, dtype=np.uint8)
        bits = self.rng.integers(0, 2, size=self.bits_per_frame, dtype=np.uint8)
        self.next_index = index + 1
        self._cached_index = index
        self._cached_bits = bits.copy()
        return bits.copy()

class Alg7KraAdaptor(FrameCryptoPlugin):
    name = 'alg7_kra'
    encode_batch_size = 4  # Unmodified speech_encoder processes four 20-ms speech frames.

    def __init__(self, algo_dir, seed=37, synth_seed=12345, output_gain=1.0):
        self.directory, m = _load_module_tree(algo_dir)
        self.original_seed = int(seed)
        self.analyzer = m['speech_analyzer'].Speech_Analyzer_class(self.directory/'speech_model.npz', self.directory/'spherical_code.npz')
        self.crypt = m['encryption'].Encryption_class(self.directory/'spherical_code.npz')
        self.ps = m['pseudo_speech'].Pseudospeech_Synthetizer_class(self.directory/'pseudospeech_model.npz', self.directory/'spherical_code.npz')
        self.PseudoState = m['pseudo_speech'].parameters_pseudospeech
        self.pitch_estimator_class = m['single_pitch'].single_pitch
        self.synth = m['DirectLPCSynthesizer'].DirectLPCSynthesizer(fs=8000,frame_size=160,periodic_mix=.5,seed=synth_seed)
        self.enc_state = self.PseudoState(self.ps.band_num,self.ps.harm_num_max)
        self.dec_state = self.PseudoState(self.ps.band_num,self.ps.harm_num_max)
        bounds = np.array([self.ps.pitch_min,self.ps.pitch_max],dtype=np.float64)/self.ps.Fs
        self.f0 = self.pitch_estimator_class(self.ps.flng,self.ps.harm_num_max,bounds)
        self.rot = np.exp(-1j*2*np.pi*np.arange(self.ps.band_num)/self.ps.band_num)
        self.bits_per_frame = int(self.crypt.encryption_bits[2])
        self._key_cursor = OriginalNumpyRngCursor(self.original_seed, self.bits_per_frame)
        self.output_gain = float(output_gain)
        if int(self.ps.Fs)!=16000 or int(self.ps.flng+self.ps.glng)!=320 or int(self.analyzer.speech_frame_size)!=160:
            raise ValueError('This adapter expects speech analyzer 8k/160, pseudo-speech 16k/320; actual: analyzer=%s, ps.Fs=%s, ps.frame=%s' % (self.analyzer.speech_frame_size, self.ps.Fs, self.ps.flng+self.ps.glng))
        self.lost_count = 0

    def get_key_for_frame(self, frame_index):
        """Return EXACT original np.random.default_rng(seed) bits for this frame.

        Forward jumps consume skipped frame draws, not a new key algorithm.
        Backward jumps reset and replay original draws from seed.
        Sequential TX and RX use separate Alg7KraAdaptor instances.
        """
        return self._key_cursor.get_key_for_frame(frame_index)

    def _keyed(self, p,e,t,idx,decrypt):
        bits = self.get_key_for_frame(idx)
        b0,b1,b2 = map(int,self.crypt.encryption_bits)
        if decrypt:
            return (self.crypt.Decrypt_translation(p,bits[:b0]),
                    self.crypt.Decrypt_translation(e,bits[b0:b1]),
                    self.crypt.Decrypt_rotation(t,bits[b1:b2]))
        return (self.crypt.Encrypt_translation(p,bits[:b0]),
                self.crypt.Encrypt_translation(e,bits[b0:b1]),
                self.crypt.Encrypt_rotation(t,bits[b1:b2]))

    def encode_batch(self, frames, indices):
        frames = check_batch(frames,indices)
        if len(frames) != self.encode_batch_size:
            raise ValueError('Legacy LPCNet TX requires batches of exactly four 20ms frames')
        x16 = np.clip(np.rint(frames.reshape(-1)*32767.0),-32768,32767).astype(np.int16)
        x8 = resample_poly(x16.astype(np.float64), 1, 2)
        x8 = np.clip(np.rint(x8),-32768,32767).astype(np.int16)
        with _analyzer_workdir(self.directory):
            p,e,t,*_ = self.analyzer.analyze_speech(x8)
        if len(p)!=len(frames):
            raise RuntimeError('speech_encoder produced %d feature frames for %d input frames' % (len(p),len(frames)))
        out=[]
        for i,abs_index in enumerate(indices):
            pe,ee,te = self._keyed(p[i],e[i],t[:,i],abs_index,False)
            y = self.ps.synthetize_frame(pe,ee,te,self.enc_state)
            if len(y)!=320:
                raise RuntimeError('PseudoSpeech generated %d samples, expected 320' % len(y))
            out.append(np.asarray(y,dtype=np.float32)/32768.0)
        return np.asarray(out,dtype=np.float32)

    def decode_frame(self,pcm,ctx):
        if ctx.status != 'ok' or pcm is None:
            self.lost_count += 1
            return np.zeros(320,dtype=np.float32)
        pcm = check_frame(pcm)
        x = np.asarray(np.clip(np.rint(pcm*32767.0),-32768,32767),dtype=np.int16)
        chunk = x[self.ps.glng:].astype(np.float64)
        X = np.fft.fft(chunk)
        self.dec_state.envelope_vector = 2*np.real((X[:len(X)//2]@self.ps.filter_bank.T)*self.rot)
        self.dec_state.energy_value = max(float(np.dot(chunk,chunk)),1e-12)
        self.dec_state.timbre_value = self.dec_state.envelope_vector / max(float(np.linalg.norm(self.dec_state.envelope_vector)),.001)
        self.dec_state.pitch_value = (self.ps.Fs/(2*np.pi))*self.f0.est(chunk,eps=2e-2)
        self.ps.value_to_index(self.dec_state)
        p,e,t = self._keyed(self.dec_state.pitch_index,self.dec_state.energy_index,self.dec_state.timbre_index,ctx.index,True)
        lo,span = float(self.analyzer.guard_low),float(self.analyzer.indices_range)
        pitch = float(self.analyzer.pitch_period_min) + ((float(p)-lo)/span)*(float(self.analyzer.pitch_period_max)-float(self.analyzer.pitch_period_min))
        energy = 10.0**(float(self.analyzer.energy_min) + ((float(e)-lo)/span)*(float(self.analyzer.energy_max)-float(self.analyzer.energy_min)))
        angle = np.asarray(t,dtype=np.float64).reshape(8)*np.sqrt(8.0)/2.0
        direction = _sphere_cart(angle)
        bands = np.maximum(energy*direction*direction,1e-12)
        cep = dct(np.log10(bands),type=2,norm='ortho')
        cep[0] -= 4.0
        lpc,g = _lpc_cep(cep)
        y8 = self.synth.synthesize_frame(pitch,np.log10(max(float(g),1e-30)),lpc)
        y16 = resample_poly(np.asarray(y8,dtype=np.float64),2,1)
        if len(y16)!=320:
            raise RuntimeError('LPC synthesized %d samples, expected 320' % len(y16))
        # DirectLPC outputs in 16-bit PCM amplitude units; no per-frame AGC.
        return np.clip(y16*self.output_gain/32768.0,-1.0,1.0).astype(np.float32)