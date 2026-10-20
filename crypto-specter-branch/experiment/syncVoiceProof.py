"""Two-terminal live VoiceProof against the EXISTING sync.v2_timepack.

No SonoBus, browser, virtual-mic routing, or external audio application.
Transport 'pulse': same-PC PulseAudio null-sink, TX->RX over its monitor.
Transport 'udp'  : real network, direct PCM16 or G.711-style mu-law at 20 kHz.
Nothing about encryption/frame index is sent as UDP metadata: only audio bytes.

NOT A SECURE ENCRYPTION ALGORITHM. Diagnostic sample permutation ONLY.
"""
import argparse
import audioop
import csv
import contextlib
import hashlib
import hmac
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

RATE, WIRE_RATE, FRAME, GUARD, SAMPLES = 16000, 20000, 320, 80, 400
LOOP_SINK = 'nears_sync_proof'
TELEMOST_TX_SINK, TELEMOST_RX_SINK = 'tx_enc_sink', 'rx_sink'


def pactl(*args):
    return subprocess.check_output(['pactl'] + list(args), text=True).strip()


def default(kind):
    field = 'Default Source:' if kind == 'sources' else 'Default Sink:'
    names = [line.split()[1] for line in pactl('list', 'short', kind).splitlines() if len(line.split()) >= 2]
    info = next((s.split(':', 1)[1].strip() for s in pactl('info').splitlines() if s.startswith(field)), '')
    allowed = [x for x in names if not x.endswith('.monitor') and not x.startswith('nears_') and x not in (LOOP_SINK, TELEMOST_TX_SINK, TELEMOST_RX_SINK, 'tx_mic')]
    if info in allowed: return info
    preferred = ['alsa_input.', 'bluez_input.'] if kind == 'sources' else ['alsa_output.', 'bluez_output.']
    return next((x for x in allowed if x.startswith(tuple(preferred))), allowed[0] if allowed else None)


def require(name, kind):
    if not name: raise RuntimeError('Cannot find physical %s: pactl list short %s' % (kind, kind))
    return name


def create_sink():
    if any(len(x.split()) > 1 and x.split()[1] == LOOP_SINK for x in pactl('list', 'short', 'sinks').splitlines()):
        return None
    out = pactl('load-module', 'module-null-sink', 'sink_name=' + LOOP_SINK,
                'sink_properties=device.description=NearS_Sync_Proof', 'rate=20000', 'channels=1', 'format=s16le')
    print('[RX] Virtual PCM loop created:', LOOP_SINK, 'module=', out, flush=True)
    return out


def read_exact(stream, count):
    parts, size = [], 0
    while size < count:
        buf = stream.read(count - size)
        if not buf: break
        parts.append(buf)
        size += len(buf)
    return b''.join(parts)


def s16(samples):
    return np.rint(np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()


def pcm(data):
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0


def make_wav(path, rate):
    w = wave.open(str(path), 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    return w


class ProofCipher:
    """Losslessly invertible 320-sample permutation keyed by ABSOLUTE frame index.

    Purpose: audible proof of correct / wrong index. No claims of security.
    """
    def __init__(self, key, session='syncproof-1'):
        self.key, self.session = key, session.encode('utf-8')
        self.last, self.plan = None, None

    def _plan(self, index):
        index = int(index)
        if index < 0 or index >= 2**64: raise ValueError('bad absolute frame index')
        if index != self.last:
            raw = hmac.new(self.key, b'NearS-DIAG-PERM-1\x00' + self.session + index.to_bytes(8, 'big'), hashlib.sha256).digest()
            rng = np.random.default_rng(int.from_bytes(raw[:16], 'big'))
            self.plan = rng.permutation(FRAME), rng.integers(0, 2, FRAME, dtype=np.uint8).astype(bool)
            self.last = index
        return self.plan

    def encrypt(self, x, index):
        perm, invert = self._plan(index)
        y = np.asarray(x, dtype=np.float32)[perm].copy()
        y[invert] *= -1
        return y

    def decrypt(self, x, index):
        perm, invert = self._plan(index)
        y = np.asarray(x, dtype=np.float32).copy()
        y[invert] *= -1
        out = np.empty(FRAME, dtype=np.float32)
        out[perm] = y
        return out


def guards():
    from sync.v2_timepack import makeGuardBank
    bank = np.asarray(makeGuardBank(WIRE_RATE, GUARD, 64), dtype=np.float32)
    if bank.shape != (64, GUARD): raise RuntimeError('wrong PN bank shape: %s' % (bank.shape,))
    return bank


def make_frame(cipher, bank, x, index, gain, guard_gain):
    enc = cipher.encrypt(x, index) * gain
    return np.concatenate((enc, bank[index % 64] * guard_gain))


def check(args):
    key = Path(args.key_file).expanduser().read_bytes()
    if len(key) < 16: raise RuntimeError('--key-file must contain at least 16 binary bytes')
    return key


def transport_sink(args):
    if args.transport == 'telemost':
        return TELEMOST_TX_SINK if args.role == 'tx' else TELEMOST_RX_SINK
    return LOOP_SINK


def live_tx(args):
    source = require(args.mic_source or default('sources'), 'sources')
    if args.transport in ('pulse','telemost'):
        tx_sink=transport_sink(args)
        if not any(x.split()[1] == tx_sink for x in pactl('list', 'short', 'sinks').splitlines() if len(x.split()) >= 2):
            raise RuntimeError('TX output virtual sink does not exist: %s' % tx_sink)
    key = check(args); cipher = ProofCipher(key, args.session); bank = guards()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    capture = subprocess.Popen(['parec', '--raw', '--format=s16le', '--rate=16000', '--channels=1',
                                '--latency-msec=40', '--device=' + source], stdout=subprocess.PIPE, bufsize=0)
    if args.transport=='telemost': args.telemost_capture=capture
    sink = None; udp = None
    if args.transport in ('pulse','telemost'):
        sink = subprocess.Popen(['paplay', '--raw', '--format=s16le', '--rate=20000', '--channels=1',
                                 '--device=' + transport_sink(args)], stdin=subprocess.PIPE, bufsize=0)
    else:
        if not args.host: raise RuntimeError('TX --transport udp requires --host RX_IP')
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n = 0; started = time.monotonic()
    print('[TX] physical microphone:', source, '| transport:', args.transport, '| session:', args.session, flush=True)
    print('[TX] No index or crypto metadata is sent: only 400 audio samples per frame.', flush=True)
    wf = make_wav(outdir/'tx_original_voice_16k.wav', RATE)
    wire_w = make_wav(outdir/'tx_encrypted_20k.wav', WIRE_RATE) if args.transport == 'telemost' else None
    try:
        while not getattr(args, 'telemost_stop', threading.Event()).is_set() and (args.duration == 0 or time.monotonic() - started < args.duration):
            data = read_exact(capture.stdout, FRAME * 2)
            if len(data) != FRAME * 2: break
            inp = pcm(data)
            packet = s16(make_frame(cipher, bank, inp, n, args.tx_gain, args.guard_amp))
            if wire_w: wire_w.writeframes(packet)
            if udp:
                if args.codec == 'ulaw': packet = audioop.lin2ulaw(packet, 2)
                udp.sendto(packet, (args.host, args.port))
            else:
                sink.stdin.write(packet)
            wf.writeframes(data); n += 1
            if n % 250 == 0:
                rms = float(np.sqrt(np.mean(inp.astype(np.float64)**2)))
                print('[TX] frame=%d audio=%.1fs mic_rms=%.4f' % (n-1,n/50,rms),flush=True)
    except (KeyboardInterrupt, BrokenPipeError):
        print('[TX] stopped',flush=True)
    finally:
        wf.close()
        if wire_w:wire_w.close()
        if udp: udp.close()
        if sink:
            try:sink.stdin.close()
            except (BrokenPipeError, OSError):pass
            try:sink.wait(timeout=3)
            except subprocess.TimeoutExpired:sink.terminate();sink.wait()
        if capture.poll() is None:capture.terminate()
        capture.wait()
        stats = {'tx_frames':n,'last_tx_frame':n-1,'audio_seconds':n/50,
                 'mic_source':source,'transport':args.transport,'codec':args.codec if udp else 'pcm16'}
        (outdir/'tx_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
        print('[TX] saved:',outdir,'frames:',n,flush=True)


def live_rx(args):
    from sync.v2_timepack import FixedGuardSyncV1, SyncV1Config
    sink = require(args.speaker_sink or default('sinks'), 'sinks')
    key = check(args)
    correct = ProofCipher(key, args.session)
    wrong = ProofCipher(key, args.session)
    outdir = Path(args.out);outdir.mkdir(parents=True,exist_ok=True)
    module=None;capture=None;sock=None;player=None;sync=None
    mode = {'value': args.mode};stop = threading.Event()
    numbers = {'frames':0,'ok':0,'lost':0,'last':None,'first':None,'resets':0}
    w_cipher=make_wav(outdir/'rx_cipher_16k.wav',RATE)
    w_correct=make_wav(outdir/'rx_correct_16k.wav',RATE)
    w_wrong=make_wav(outdir/'rx_wrong_16k.wav',RATE)
    w_wire=make_wav(outdir/'rx_received_20k.wav',WIRE_RATE) if args.transport=='telemost' else None
    csvfile=(outdir/'frames.csv').open('w',newline='')
    events=(outdir/'events.jsonl').open('w')
    writer=csv.writer(csvfile);writer.writerow(['index','status','guard_pos','corr'])

    def reader_commands():
        while not stop.is_set():
            ready,_,_ = select.select([sys.stdin],[],[],0.3)
            if not ready: continue
            line = sys.stdin.readline().strip().lower()
            if not line:
                if args.transport=='telemost':
                    print('[RX] ENTER: stopping and saving WAV/CSV...',flush=True)
                    stop.set()
                    if capture and capture.poll() is None: capture.terminate()
                    return
                time.sleep(0.5)
                continue
            if line in ('1','2','3'):
                mode['value'] = {'1':'raw','2':'correct','3':'wrong'}[line]
                print('[RX] SWITCHED TO:',mode['value'],flush=True)
            elif line == 'q':
                stop.set()
                if capture and capture.poll() is None: capture.terminate()

    def on_frame(index, core, status, meta):
        index=int(index)
        x=np.asarray(core,dtype=np.float32)[:FRAME] if status=='ok' else np.zeros(FRAME,np.float32)
        if len(x)!=FRAME:raise ValueError('sync returned %d samples, expected 320'%len(x))
        x=np.clip(x/args.tx_gain,-1,1)
        if status=='ok': numbers['ok']+=1
        else:numbers['lost']+=1
        y=correct.decrypt(x,index)
        z=wrong.decrypt(x,index+args.wrong_shift)
        w_cipher.writeframes(s16(x));w_correct.writeframes(s16(y));w_wrong.writeframes(s16(z))
        current=mode['value'];v={'raw':x,'correct':y,'wrong':z}[current]
        try:player.stdin.write(s16(v))
        except BrokenPipeError:stop.set()
        writer.writerow([index,status,meta.get('guard_pos',''),meta.get('corr','')])
        numbers['frames']+=1;numbers['last']=index
        if numbers['first'] is None:numbers['first']=index
        if numbers['frames']%250==0:
            print('[RX] frame=%d good=%d lost=%d monitor=%s' % (index,numbers['ok'],numbers['lost'],mode['value']),flush=True)

    def on_event(event):
        events.write(json.dumps(event,default=str)+'\n');events.flush()
        if event.get('event') in ('acquired','reacquired','lost','track_miss'):
            print('[SYNC EVENT]',event,flush=True)

    try:
        if args.transport == 'pulse':module=create_sink()
        player=subprocess.Popen(['paplay','--raw','--format=s16le','--rate=16000','--channels=1',
                                  '--device='+sink],stdin=subprocess.PIPE,bufsize=0)
        if args.transport in ('pulse','telemost'):
            capture=subprocess.Popen(['parec','--raw','--format=s16le','--rate=20000',
                                      '--channels=1','--latency-msec=40','--device='+transport_sink(args)+'.monitor'],stdout=subprocess.PIPE,bufsize=0)
        else:
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind((args.bind,args.port));sock.settimeout(0.3)
        cfg=SyncV1Config(rate=WIRE_RATE,frameSamples=SAMPLES,coreSamples=FRAME,
                         guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8)
        sync=FixedGuardSyncV1(cfg,on_frame,on_event)
        threading.Thread(target=reader_commands,daemon=True).start()
        print('[RX] transport=%s speaker=%s codec=%s listening=%s:%s' % (
            args.transport,sink,args.codec,args.bind,args.port),flush=True)
        print('[RX] CONTROLS: 1=RAW ENCRYPTED, 2=CORRECT, 3=WRONG(+%d), q=QUIT' % args.wrong_shift,flush=True)
        if args.transport in ('pulse','telemost'):print('[RX] NOW START TX on the other PC.',flush=True)
        started=time.monotonic()
        # Deliberately re-chunk the audio stream into chunks not aligned to 400 samples.
        # The synchronizer sees raw continuous PCM and MUST find PN positions itself.
        pending=np.empty(0,dtype=np.float32)
        while not stop.is_set() and (args.duration==0 or time.monotonic()-started<args.duration):
            if sock:
                try:data,_=sock.recvfrom(4096)
                except socket.timeout:continue
                expected=400 if args.codec=='ulaw' else 800
                if len(data)!=expected:continue
                if args.codec=='ulaw':data=audioop.ulaw2lin(data,2)
            else:
                data=read_exact(capture.stdout,800)
                if len(data)!=800:break
            if w_wire:w_wire.writeframes(data)
            pending=np.concatenate((pending,pcm(data)))
            while len(pending)>=173:
                sync.push(pending[:173],time.monotonic())
                pending=pending[173:]
        if len(pending):sync.push(pending,time.monotonic())
        print('[RX] sync state:',getattr(sync,'state','unknown'),flush=True)
        print('[RX] sync.stats:',getattr(sync,'stats',{}),flush=True)
    except KeyboardInterrupt:
        print('[RX] interrupted',flush=True)
    finally:
        stop.set()
        if sync is not None:
            numbers['sync_stats']=dict(getattr(sync,'stats',{}))
            numbers['sync_state']=str(getattr(sync,'state','unknown'))
        numbers['transport']=args.transport
        for w in (w_cipher,w_correct,w_wrong):w.close()
        if w_wire:w_wire.close()
        csvfile.close();events.close()
        if player:
            try:player.stdin.close()
            except (BrokenPipeError,OSError):pass
            try:player.wait(timeout=3)
            except subprocess.TimeoutExpired:player.terminate();player.wait()
        if capture:
            if capture.poll() is None:capture.terminate()
            capture.wait()
        if sock:sock.close()
        if module:
            try:pactl('unload-module',module)
            except subprocess.CalledProcessError:pass
        (outdir/'rx_summary.json').write_text(json.dumps(numbers,indent=2)+'\n')
        print('[RX] saved 3 WAVs + frames.csv in',outdir,flush=True)


def telemost_tx(args):
    """Same verified ProofCipher and live_tx, only virtual Chrome routing differs."""
    from experiment.telemostTest import VirtualAudio, AutoRouter
    # Select physical MIC *before* virtual source creation, otherwise PA defaults can change.
    args.mic_source=require(args.mic_source or default('sources'),'sources')
    print('[TX] Physical microphone:',args.mic_source,flush=True)
    with VirtualAudio('tx'):
        print('Telemost TX: choose MICROPHONE NearS_TX_Microphone in Chrome.',flush=True)
        input('PRESS ENTER to START live encrypted speech... ')
        with AutoRouter('tx') as router:
            router.ensure_tx()
            args.telemost_stop=threading.Event()
            def wait_stop():
                try:input('[TX] Sending. PRESS ENTER to STOP and save... ')
                except EOFError:return
                args.telemost_stop.set()
                proc=getattr(args,'telemost_capture',None)
                if proc and proc.poll() is None:proc.terminate()
            threading.Thread(target=wait_stop,daemon=True).start()
            live_tx(args)
    print('[TX] Finished; stop RX after 2 seconds.',flush=True)


def telemost_rx(args):
    from experiment.telemostTest import VirtualAudio, AutoRouter
    # Select real headphones before PA virtual RX sink exists.
    args.speaker_sink=require(args.speaker_sink or default('sinks'),'sinks')
    print('[RX] Physical headphones:',args.speaker_sink,flush=True)
    with VirtualAudio('rx'):
        print('Telemost RX: choose SPEAKERS NearS_RX in Chrome.',flush=True)
        input('PRESS ENTER to START receiving audio... ')
        with AutoRouter('rx'):
            print('[RX] Press 1+Enter RAW, 2+Enter CORRECT, 3+Enter WRONG; empty ENTER stops.',flush=True)
            live_rx(args)


def selftest(args):
    key=check(args); a=ProofCipher(key,args.session); b=ProofCipher(key,args.session)
    c=ProofCipher(key,args.session);rng=np.random.default_rng(456)
    samples=rng.normal(0,.11,(128,FRAME)).astype(np.float32)
    good=np.stack([b.decrypt(a.encrypt(x,i),i) for i,x in enumerate(samples)])
    wrong=np.stack([c.decrypt(a.encrypt(x,i),i+64) for i,x in enumerate(samples)])
    assert np.array_equal(samples,good)
    assert abs(np.corrcoef(samples.ravel(),wrong.ravel())[0,1])<.1
    print('PASS: frameIndex N decodes EXACTLY; frameIndex N+64 produces different PCM',flush=True)
    from sync.v2_timepack import FixedGuardSyncV1,SyncV1Config
    bank=guards();recovered=[];events=[]
    def on_frame(i,x,status,meta):recovered.append((int(i),status,np.asarray(x,dtype=np.float32).copy()))
    sync=FixedGuardSyncV1(SyncV1Config(rate=WIRE_RATE,frameSamples=SAMPLES,coreSamples=FRAME,
                       guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8),on_frame,events.append)
    wire=np.concatenate([make_frame(a,bank,samples[i],i,.75,.12) for i in range(len(samples))])
    for start in range(0,len(wire),173):sync.push(wire[start:start+173],time.monotonic())
    sync.finish()
    inds=[i for i,status,_ in recovered if status=='ok']
    if inds!=list(range(len(samples))):
        raise AssertionError('SYNC NOT EXACT 0..127: %s ... %s' % (inds[:12],inds[-12:]))
    corr=np.stack([b.decrypt(x[:FRAME]/.75,i) for i,status,x in recovered if status=='ok'])
    assert np.allclose(samples,corr,atol=2e-6)
    print('PASS: loaded sync.v2_timepack receives 128/128 PN frames with exact indices, decrypts correctly',flush=True)


def main():
    p=argparse.ArgumentParser(description='NearS LIVE 3-mode proof: actual sync.v2_timepack, local PulseAudio, UDP or Telemost')
    p.add_argument('--role',choices=['tx','rx','selftest'],required=True)
    p.add_argument('--transport',choices=['pulse','udp','telemost'],default='pulse')
    p.add_argument('--key-file',default='.secrets/stft_test.key')
    p.add_argument('--session',default='syncproof-1')
    p.add_argument('--out',default='/tmp/syncproof')
    p.add_argument('--mode',choices=['raw','correct','wrong'],default='correct')
    p.add_argument('--wrong-shift',type=int,default=64)
    p.add_argument('--duration',type=float,default=0,help='seconds, 0=until Ctrl+C')
    p.add_argument('--mic-source')
    p.add_argument('--speaker-sink')
    p.add_argument('--host',help='TX --transport udp: RX IP')
    p.add_argument('--bind',default='0.0.0.0')
    p.add_argument('--port',type=int,default=19333)
    p.add_argument('--codec',choices=['pcm16','ulaw'],default='pcm16',help='UDP only. ulaw uses G.711 companding at 20k, NOT canonical 8k G.711')
    p.add_argument('--tx-gain',type=float,default=.75)
    p.add_argument('--guard-amp',type=float,default=.12)
    args=p.parse_args()
    if args.tx_gain<=0 or args.guard_amp<=0 or args.duration<0:p.error('invalid gain or duration')
    if args.role=='tx':
        telemost_tx(args) if args.transport=='telemost' else live_tx(args)
    elif args.role=='rx':
        telemost_rx(args) if args.transport=='telemost' else live_rx(args)
    else:selftest(args)

if __name__=='__main__':main()