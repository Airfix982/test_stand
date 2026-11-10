#!/usr/bin/env python3
"""ALG7 -> TimePack 320+80 -> VDO.Ninja -> SuperSpinner -> ALG7, ONE PC / two terminals."""
import argparse, csv, json, os, re, select, shutil, subprocess as sp, sys, threading, time, wave
from pathlib import Path
for v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import numpy as np
from crypto.alg7_kra_adaptor import Alg7KraAdaptor, OriginalNumpyRngCursor
from crypto.frame_port import FrameContext
from experiment.telemostTest import VirtualAudio, pactl, TX_SINK, RX_SINK
from sync.v2_timepack import FixedGuardSyncV1, SyncV1Config, makeGuardBank

FS, WFS, CORE, GUARD, FRAME = 16000, 20000, 320, 80, 400
STREAM = 'nearsalg7onepc20261021'

def pcm(x): return np.rint(np.clip(x, -1., 1.)*32767).astype('<i2').tobytes()
def unpack(x): return np.frombuffer(x, '<i2').astype(np.float32)/32768.
def wav(path, rate):
    w=wave.open(str(path),'wb'); w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);return w

def exact(stream, size):
    buf=bytearray()
    while len(buf)<size:
        b=stream.read(size-len(buf))
        if not b: break
        buf.extend(b)
    return bytes(buf)

def physical(kind, explicit):
    names=[x.split()[1] for x in pactl('list','short',kind).splitlines() if len(x.split())>1]
    default_tag='Default Source:' if kind=='sources' else 'Default Sink:'
    d=next((x.split(':',1)[1].strip() for x in pactl('info').splitlines() if x.startswith(default_tag)),None)
    x=explicit or d
    if x in names and not x.endswith('.monitor') and x not in ('tx_mic','tx_enc_sink','rx_sink'):return x
    options=[v for v in names if v.startswith(('alsa_','bluez_')) and not v.endswith('.monitor')]
    if options:return options[0]
    raise RuntimeError('Select physical %s with --mic / --speaker; pactl list short %s' %(kind,kind))

def process(argv, input=False):
    return sp.Popen(argv,stdin=sp.PIPE if input else None,stdout=None if input else sp.PIPE,bufsize=0)

def done(p, close_stdin=False):
    if not p:return
    if close_stdin and p.stdin:
        try:p.stdin.close()
        except (BrokenPipeError,OSError):pass
        try:p.wait(timeout=2)
        except sp.TimeoutExpired:pass
    if p.poll() is None:p.terminate()
    try:p.wait(timeout=2)
    except sp.TimeoutExpired:p.kill();p.wait()

def codec(a): return Alg7KraAdaptor(str(Path(a.algo_dir).resolve()),seed=a.seed,output_gain=a.output_gain)
def url(a):
    base='https://vdo.ninja/?'
    if a.role=='tx':return base+'push='+a.stream+'&miconly&aec=0&agc=0&denoise=0&noap'
    return base+'view='+a.stream+'&audiocodec=pcm&novideo&noap'

def browser(a):
    exe=next((shutil.which(x) for x in ('google-chrome','google-chrome-stable','chromium','chromium-browser') if shutil.which(x)),None)
    if not exe:raise RuntimeError('Chrome/Chromium missing; open URL manually: '+url(a))
    # Independent profiles => separate Chrome processes and audio streams.
    profile=Path.home()/('.cache/nears_vdo_alg7_'+a.role)
    sp.Popen([exe,'--user-data-dir='+str(profile),'--no-first-run','--new-window',url(a)],
             stdout=sp.DEVNULL,stderr=sp.DEVNULL)
    print('[BROWSER]',url(a),'profile=',profile,flush=True)
    return str(profile)

def belongs(pid, profile):
    for _ in range(15):
        try:
            raw=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode('utf-8','ignore')
            if '--user-data-dir='+profile in raw:return True
            lines=(Path('/proc')/str(pid)/'status').read_text().splitlines()
            pid=int(next(x.split(':',1)[1].strip() for x in lines if x.startswith('PPid:')))
            if pid<2:return False
        except (OSError,ValueError,StopIteration):return False
    return False

def route_rx(stop, profile):
    """Only route playback from RX Chrome PROFILE; never touch TX Chrome."""
    seen=set()
    while not stop.wait(.7):
        try:
            for b in re.split(r'(?=Sink Input #\d+)',pactl('list','sink-inputs')):
                match=re.search(r'Sink Input #(\d+)',b);pid=re.search(r'application.process.id = "(\d+)"',b)
                if not (match and pid and 'application.process.binary = "chrome"' in b):continue
                sid=match.group(1)
                if belongs(pid.group(1),profile):
                    sp.run(['pactl','move-sink-input',sid,RX_SINK],stdout=sp.DEVNULL,stderr=sp.DEVNULL)
                    if sid not in seen:print('[RX] Browser audio routed to NearS_RX, stream #'+sid,flush=True);seen.add(sid)
        except Exception as e:print('[RX routing]',str(e),flush=True)

def keyboard(stop, role, mode, cap):
    while not stop.is_set():
        ready,_,_=select.select([sys.stdin],[],[],.2)
        if not ready:continue
        s=sys.stdin.readline()
        if not s:return
        s=s.strip().lower()
        if role=='rx' and s in ('1','2','3'):
            mode[0]={'1':'raw','2':'correct','3':'wrong'}[s];print('[RX] MODE:',mode[0],flush=True)
        elif not s or s=='q':
            stop.set()
            if cap.poll() is None:cap.terminate()
            return

def sender(a):
    source=physical('sources',a.mic);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    alg=codec(a);bank=np.asarray(makeGuardBank(WFS,GUARD,64),np.float32)
    stop=threading.Event();rec=play=None;count=0;cpu=0.;beg=None
    try:
        with VirtualAudio('tx'):
            browser(a)
            print('TX: Chrome -> MICROPHONE NearS_TX_Microphone -> START. RX must be running.',flush=True)
            input('TX: Press ENTER to transmit LIVE speech... ')
            rec=process(['parec','--raw','--format=s16le','--channels=1','--rate=16000','--device='+source]);
            play=process(['paplay','--raw','--format=s16le','--channels=1','--rate=20000','--device='+TX_SINK],True)
            threading.Thread(target=keyboard,args=(stop,'tx',None,rec),daemon=True).start()
            print('TX STARTED. ENTER stops.',flush=True);beg=time.monotonic()
            with wav(out/'tx_original_16k.wav',FS) as src, wav(out/'tx_encrypted_20k.wav',WFS) as enc:
                while not stop.is_set():
                    b=exact(rec.stdout,CORE*2*alg.encode_batch_size)
                    if len(b)!=CORE*2*alg.encode_batch_size or stop.is_set():break
                    x=unpack(b).reshape(alg.encode_batch_size,CORE)
                    indices=np.arange(count,count+len(x),dtype=np.int64)
                    t=time.process_time();encrypted=np.asarray(alg.encode_batch(x,indices),np.float32);cpu+=time.process_time()-t
                    if encrypted.shape!=x.shape:raise RuntimeError('ALG7 wrong shape '+str(encrypted.shape))
                    frames=[pcm(np.r_[encrypted[i]*a.tx_gain,bank[int(idx)%64]*a.guard_amp]) for i,idx in enumerate(indices)]
                    wire=b''.join(frames);src.writeframes(b);enc.writeframes(wire);play.stdin.write(wire)
                    count+=len(x)
                    if count%200==0:print('[TX] frames=%d CPU=%.2fx'%(count,cpu/(count/50)),flush=True)
                    if a.duration and time.monotonic()-beg>=a.duration:break
    except (KeyboardInterrupt,BrokenPipeError):pass
    finally:
        stop.set();done(rec);done(play,True);alg.close()
        (out/'summary.json').write_text(json.dumps({'frames':count,'encode_cpu_sec':cpu},indent=2))
        print('TX saved:',out,flush=True)

def receiver(a):
    speaker=physical('sinks',a.speaker);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    alg=codec(a);rec=play=None;stop=threading.Event();mode=[a.mode];sync=None
    stats={'ok':0,'lost':0,'decoded':0,'rx_samples':0,'dec_cpu_sec':0.,'sync_cpu_sec':0.}
    try:
        with VirtualAudio('rx'):
            profile=browser(a)
            print('RX: Chrome playback must go to NearS_RX; use pavucontrol > Playback if auto-route fails.',flush=True)
            input('RX: Press ENTER to START recording... ')
            play=process(['paplay','--raw','--format=s16le','--channels=1','--rate=16000','--device='+speaker],True)
            rec=process(['parec','--raw','--format=s16le','--channels=1','--rate=20000','--device='+RX_SINK+'.monitor'])
            threading.Thread(target=route_rx,args=(stop,profile),daemon=True).start()
            threading.Thread(target=keyboard,args=(stop,'rx',mode,rec),daemon=True).start()
            print('RX: 1+ENTER=RAW  2+ENTER=CORRECT  3+ENTER=WRONG  ENTER=STOP',flush=True)
            start=time.monotonic()
            with wav(out/'rx_incoming_20k.wav',WFS) as incoming, wav(out/'rx_raw_16k.wav',FS) as raw, \
                 wav(out/'rx_listened_16k.wav',FS) as listened, (out/'frames.csv').open('w',newline='') as f, \
                 (out/'events.jsonl').open('w') as ef:
                w=csv.writer(f);w.writerow(['frame','status','corr','mode','decode_ms'])
                def frame(idx,packet,status,meta):
                    idx=int(idx);valid=status=='ok';stats['ok' if valid else 'lost']+=1
                    cipher=np.clip(np.asarray(packet,np.float32)[:CORE]/a.tx_gain,-1,1) if valid else np.zeros(CORE,np.float32)
                    raw.writeframes(pcm(cipher));m=mode[0];t=time.process_time();start_dec=time.perf_counter()
                    if m=='raw':y=cipher
                    elif not valid:y=np.zeros(CORE,np.float32)
                    else:
                        y=alg.decode_frame(cipher,FrameContext(idx+(a.wrong_shift if m=='wrong' else 0),'ok'))
                        stats['decoded']+=1
                    stats['dec_cpu_sec']+=time.process_time()-t
                    y=np.asarray(y,np.float32).reshape(CORE);b=pcm(y);listened.writeframes(b);play.stdin.write(b)
                    w.writerow([idx,status,meta.get('corr',''),m,round((time.perf_counter()-start_dec)*1000,2)])
                    if (stats['ok']+stats['lost'])%200==0:print('[RX] frame=%d ok=%d lost=%d mode=%s'%(idx,stats['ok'],stats['lost'],m),flush=True)
                def event(ev):
                    alg.on_sync_event(ev);ef.write(json.dumps(ev,default=str)+'\n');ef.flush()
                    if ev.get('event') in ('acquired','reacquired','track_gap_recovered','track_miss'):print('[SYNC]',ev['event'],flush=True)
                sync=FixedGuardSyncV1(SyncV1Config(rate=WFS,frameSamples=FRAME,coreSamples=CORE,guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8),frame,event)
                while not stop.is_set():
                    b=exact(rec.stdout,400)
                    if len(b)!=400:break
                    incoming.writeframes(b);x=unpack(b);stats['rx_samples']+=len(x)
                    t=time.process_time();sync.push(x,time.monotonic());stats['sync_cpu_sec']+=time.process_time()-t
                    if a.duration and time.monotonic()-start>=a.duration:break
    except (KeyboardInterrupt,BrokenPipeError):pass
    finally:
        stop.set();done(rec);done(play,True);alg.close()
        if sync:stats.update(sync_stats=sync.stats,final_state=sync.state,last_frame=sync.lastFrame)
        (out/'summary.json').write_text(json.dumps(stats,indent=2))
        print('RX saved:',out,flush=True)

def selftest(a):
    rng=np.random.default_rng(a.seed);cursor=OriginalNumpyRngCursor(a.seed,27)
    for i in range(80):assert np.array_equal(rng.integers(0,2,27,dtype=np.uint8),cursor.get_key_for_frame(i))
    bank=np.asarray(makeGuardBank(WFS,GUARD,64),np.float32);x=np.random.default_rng(99).normal(0,.1,(128,CORE)).astype(np.float32)
    results=[];s=FixedGuardSyncV1(SyncV1Config(rate=WFS,frameSamples=FRAME,coreSamples=CORE,guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8),
       lambda i,p,st,mt:results.append((i,st,np.asarray(p)[:CORE])),lambda e:None)
    signal=np.concatenate([np.r_[q*.75,bank[i%64]*.12] for i,q in enumerate(x)])
    for off in range(0,len(signal),173):s.push(signal[off:off+173],time.monotonic())
    s.finish();assert len(results)==128 and all(i==j and st=='ok' for j,(i,st,p) in enumerate(results))
    assert np.allclose(x,np.stack([p for i,st,p in results])/.75,atol=1e-5)
    print('PASS: RNG + 128/128 SuperSpinner frames (ALG7 models validated on actual TX/RX)')

def main():
    p=argparse.ArgumentParser();p.add_argument('--role',required=True,choices=['tx','rx','selftest'])
    p.add_argument('--algo-dir',default=str(Path(__file__).resolve().parent));p.add_argument('--stream',default=STREAM)
    p.add_argument('--seed',type=int,default=37);p.add_argument('--wrong-shift',type=int,default=64)
    p.add_argument('--mode',default='correct',choices=['raw','correct','wrong']);p.add_argument('--duration',type=float,default=0)
    p.add_argument('--out',default='/tmp/alg7_vdo');p.add_argument('--output-gain',type=float,default=1.)
    p.add_argument('--tx-gain',type=float,default=.75);p.add_argument('--guard-amp',type=float,default=.12)
    p.add_argument('--mic');p.add_argument('--speaker');a=p.parse_args()
    if a.role=='tx':sender(a)
    elif a.role=='rx':receiver(a)
    else:selftest(a)

if __name__=='__main__':main()