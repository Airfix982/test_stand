#!/usr/bin/env python3
"""Shared VDO.Ninja transport: one cipher adapter per dedicated algoss package."""
import argparse, json, os, re, select, shutil, subprocess as sp, sys, threading, time, wave
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'1')
import numpy as np
N = 320

def pcm(x):
    return np.clip(np.rint(np.clip(np.asarray(x,dtype=np.float64),-1.,1.)*32768.),-32768,32767).astype('<i2')

def floats(x):
    return np.asarray(x,dtype='<i2').astype(np.float32)/32768.
from experiment.telemostTest import VirtualAudio, pactl, TX_SINK, RX_SINK
from sync.v2_timepack import FixedGuardSyncV1, SyncV1Config, makeGuardBank
RATE,FS,GUARD,FRAME=20000,16000,80,400

def wav(path,rate):
    w=wave.open(str(path),'wb');w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);return w

def readn(f,n):
    data=bytearray()
    while len(data)<n:
        b=f.read(n-len(data))
        if not b:break
        data+=b
    return bytes(data)

def proc(argv,writer=False):
    return sp.Popen(argv,stdin=sp.PIPE if writer else None,stdout=None if writer else sp.PIPE,bufsize=0)

def terminate(p):
    if not p:return
    if p.stdin:
        try:p.stdin.close()
        except OSError:pass
    if p.poll() is None:p.terminate()
    try:p.wait(timeout=1)
    except sp.TimeoutExpired:p.kill();p.wait()

def device(kind,override):
    found=[s.split()[1] for s in pactl('list','short',kind).splitlines() if len(s.split())>1]
    if override:
        if override not in found:raise RuntimeError('Unknown device '+override+'; available: '+str(found))
        return override
    label='Default Source:' if kind=='sources' else 'Default Sink:'
    default=next((x.split(':',1)[1].strip() for x in pactl('info').splitlines() if x.startswith(label)),None)
    if default in found and not default.endswith('.monitor') and default not in (TX_SINK,RX_SINK,'tx_mic'):return default
    opts=[x for x in found if x.startswith(('alsa_','bluez_')) and not x.endswith('.monitor')]
    if not opts:raise RuntimeError('No physical device. Use --mic or --speaker. pactl list short '+kind)
    return opts[0]

def link(a):
    base='https://vdo.ninja/?'
    if a.role=='tx':return base+'push='+a.stream+'&miconly&aec=0&agc=0&denoise=0&noap'
    return base+'view='+a.stream+'&audiocodec=pcm&novideo&noap'

def chrome(a):
    url=link(a);print('BROWSER:',url,flush=True)
    profile=str(Path.home()/('.cache/nears_cipher_'+a.name+'_'+a.role))
    if not a.no_browser:
        exe=next((shutil.which(t) for t in ('google-chrome','google-chrome-stable','chromium','chromium-browser') if shutil.which(t)),None)
        if exe:sp.Popen([exe,'--user-data-dir='+profile,'--no-first-run','--new-window',url],stdout=sp.DEVNULL,stderr=sp.DEVNULL)
        else:print('Chrome unavailable: open URL manually (use separate profiles).',flush=True)
    return profile

def from_profile(pid,profile):
    try:
        for _ in range(16):
            if pid<=1:return False
            cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode('utf-8','ignore')
            if '--user-data-dir='+profile in cmd:return True
            text=(Path('/proc')/str(pid)/'status').read_text()
            pid=int(re.search(r'^PPid:\s*(\d+)',text,re.M).group(1))
    except (OSError,ValueError,AttributeError):pass
    return False

def route_rx(stop,profile):
    routed=set()
    while not stop.wait(.8):
        try:
            data=pactl('list','sink-inputs')
            for block in re.split(r'(?=Sink Input #\d+)',data):
                obj=re.search(r'Sink Input #(\d+)',block);pid=re.search(r'application.process.id = "(\d+)"',block)
                if not (obj and pid and from_profile(int(pid.group(1)),profile)):continue
                sid=obj.group(1)
                sp.run(['pactl','move-sink-input',sid,RX_SINK],stdout=sp.DEVNULL,stderr=sp.DEVNULL)
                if sid not in routed:print('RX routed Chrome stream #'+sid+' to NearS_RX',flush=True);routed.add(sid)
        except Exception as e:print('Routing:',str(e),flush=True)

def keys(stop,role,mode,capture):
    while not stop.is_set():
        r,_,_=select.select([sys.stdin],[],[],.2)
        if not r:continue
        s=sys.stdin.readline().strip().lower()
        if role=='rx' and s in ('1','2','3'):
            mode[0]={'1':'raw','2':'correct','3':'wrong'}[s];print('RX mode =',mode[0],flush=True)
        elif not s or s=='q':
            stop.set()
            if capture.poll() is None:capture.terminate()
            return

def tx(a):
    source=device('sources',a.mic);c=a.cipher(a.secret);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    bank=makeGuardBank(RATE,GUARD,64);stop=threading.Event();cap=play=None;num=0;total=0
    try:
        with VirtualAudio('tx'):
            chrome(a);input('TX: In Chrome select NearS_TX_Microphone, START, then press ENTER here: ')
            cap=proc(['parec','--raw','--format=s16le','--channels=1','--rate=16000','--device='+source])
            play=proc(['paplay','--raw','--format=s16le','--channels=1','--rate=20000','--device='+TX_SINK],True)
            threading.Thread(target=keys,args=(stop,'tx',None,cap),daemon=True).start()
            print('TX running. Empty ENTER stops.',flush=True)
            start=time.monotonic()
            with wav(out/'tx_original_16k.wav',FS) as raw, wav(out/'tx_cipher_20k.wav',RATE) as cipher:
                while not stop.is_set():
                    b=readn(cap.stdout,N*2)
                    if len(b)<N*2 or stop.is_set():break
                    x=floats(np.frombuffer(b,dtype='<i2'))
                    t=time.perf_counter();y=c.encode(x,num);total+=time.perf_counter()-t
                    if y.shape!=(N,):raise ValueError(f'{a.name} returned {y.shape} not (320,)')
                    frame=np.r_[y,bank[num%64]*a.guard_amp]
                    wire=pcm(frame).tobytes()
                    raw.writeframes(b);cipher.writeframes(wire);play.stdin.write(wire)
                    num+=1
                    if num%100==0:print('TX frames',num,'avg cipher ms',round(1000*total/num,2),flush=True)
                    if a.duration and time.monotonic()-start>a.duration:break
    except (KeyboardInterrupt,BrokenPipeError):pass
    finally:
        stop.set();terminate(cap);terminate(play);c.close()
        (out/'summary.json').write_text(json.dumps(dict(algo=a.name,frames=num,avg_encode_ms=total/max(num,1)*1000),indent=2))
        print('TX saved',out,flush=True)

def rx(a):
    target=device('sinks',a.speaker);c=a.cipher(a.secret);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    stop=threading.Event();mode=[a.mode];cap=play=None;sync=None
    stats=dict(ok=0,lost=0,decoded=0,decode_errors=0,decode_seconds=0.,samples=0)
    try:
        with VirtualAudio('rx'):
            profile=chrome(a)
            threading.Thread(target=route_rx,args=(stop,profile),daemon=True).start()
            input('RX: Browser audio must play INTO NearS_RX (pavucontrol if needed). Press ENTER: ')
            cap=proc(['parec','--raw','--format=s16le','--channels=1','--rate=20000','--device='+RX_SINK+'.monitor'])
            play=proc(['paplay','--raw','--format=s16le','--channels=1','--rate=16000','--device='+target],True)
            threading.Thread(target=keys,args=(stop,'rx',mode,cap),daemon=True).start()
            print('RX: 1 ENTER=RAW, 2 ENTER=CORRECT, 3 ENTER=WRONG, empty ENTER=STOP',flush=True)
            with wav(out/'rx_incoming_20k.wav',RATE) as raw, wav(out/'rx_cipher_16k.wav',FS) as cy, wav(out/'rx_listened_16k.wav',FS) as listened, (out/'frames.csv').open('w') as logfile:
                logfile.write('frame,status,mode,decode_ms\n');start=time.monotonic()
                def on_frame(index,packet,status,meta):
                    index=int(index);ok=status=='ok';stats['ok' if ok else 'lost']+=1
                    wire=np.asarray(packet,dtype=np.float32)[:N] if ok else np.zeros(N,np.float32)
                    cy.writeframes(pcm(wire).tobytes())
                    t=time.perf_counter();current=mode[0]
                    if current=='raw':voice=wire
                    elif not ok:voice=np.zeros(N,np.float32)
                    else:
                        try:
                            voice=c.decode(wire,index+(a.wrong_shift if current=='wrong' else 0))
                            stats['decoded']+=1
                        except Exception as e:
                            stats['decode_errors']+=1;voice=np.zeros(N,np.float32)
                            if stats['decode_errors']<=5:print('DECODER FAILED at frame',index,repr(e),flush=True)
                    ms=(time.perf_counter()-t)*1000;stats['decode_seconds']+=ms/1000
                    b=pcm(voice).tobytes();listened.writeframes(b);play.stdin.write(b)
                    logfile.write(f'{index},{status},{current},{ms:.3f}\n')
                    if (stats['ok']+stats['lost'])%100==0:print('RX ok=%d lost=%d mode=%s'%(stats['ok'],stats['lost'],current),flush=True)
                sync=FixedGuardSyncV1(SyncV1Config(rate=RATE,frameSamples=FRAME,coreSamples=N,guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8),on_frame,lambda e:None)
                while not stop.is_set():
                    b=readn(cap.stdout,800)  # 400x16-bit samples
                    if len(b)<800:break
                    raw.writeframes(b);x=floats(np.frombuffer(b,dtype='<i2'))
                    sync.push(x,time.monotonic());stats['samples']+=400
                    if a.duration and time.monotonic()-start>a.duration:break
    except (KeyboardInterrupt,BrokenPipeError):pass
    finally:
        stop.set();terminate(cap);terminate(play);c.close()
        if sync:stats.update(sync_stats=sync.stats,final_state=sync.state,last_frame=sync.lastFrame)
        (out/'summary.json').write_text(json.dumps(stats,indent=2,default=str))
        print('RX saved',out,flush=True)

def selftest(a):
    c=a.cipher(a.secret);bank=makeGuardBank(RATE,GUARD,64)
    rng=np.random.default_rng(42);voice=rng.normal(0,.08,(64,N)).astype(np.float32)
    result=[];s=FixedGuardSyncV1(SyncV1Config(rate=RATE,frameSamples=FRAME,coreSamples=N,guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8),lambda i,p,status,meta:result.append((int(i),status,np.asarray(p)[:N].copy())),lambda e:None)
    start=time.monotonic();packets=[]
    for i,x in enumerate(voice):packets.append(pcm(np.r_[c.encode(x,i),bank[i%64]*.12]).astype(np.int16))
    signal=floats(np.concatenate(packets))
    for offset in range(0,len(signal),173):s.push(signal[offset:offset+173],time.monotonic())
    s.finish();assert len(result)==64 and all(i==j and status=='ok' for j,(i,status,p) in enumerate(result)),[(i,status) for i,status,_ in result[:10]]
    errors=[float(np.mean((c.decode(p,i)-voice[i])**2)) for i,_,p in result]
    print('PASS',a.name,'64/64 SuperSpinner','RMSE',round(float(np.mean(errors))**.5,5),'seconds',round(time.monotonic()-start,2))

def main(cipher, name):
    p=argparse.ArgumentParser()
    p.add_argument('--role',required=True,choices=('tx','rx','selftest'))
    p.add_argument('--secret',default=os.getenv('NEARS_ALGO_SECRET','nears-test'))
    p.add_argument('--stream',default='nears_'+name+'_solo_20261024');p.add_argument('--mode',choices=('raw','correct','wrong'),default='correct')
    p.add_argument('--wrong-shift',type=int,default=64);p.add_argument('--guard-amp',type=float,default=.12)
    p.add_argument('--mic');p.add_argument('--speaker');p.add_argument('--no-browser',action='store_true')
    p.add_argument('--duration',type=float,default=0);p.add_argument('--out',default='/tmp/sync_algorithms')
    a=p.parse_args();a.cipher=cipher;a.name=name
    if a.role=='tx':tx(a)
    elif a.role=='rx':rx(a)
    else:selftest(a)

