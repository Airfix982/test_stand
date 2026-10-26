"""Service-neutral SuperSpinner transport. Python 3.8+, PulseAudio/PipeWire.

TX -> system virtual microphone NearS_TX_Microphone (source tx_mic)
RX <- system virtual speaker NearS_RX (sink rx_sink)
No browser/app auto start or automatic stream routing.
"""
import argparse, csv, json, os, select, shutil, subprocess as sp, sys, threading, time, wave
from pathlib import Path
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import numpy as np
from experiment.telemostTest import VirtualAudio, pactl, TX_SINK, RX_SINK
from sync.v2_timepack import FixedGuardSyncV1, SyncV1Config, makeGuardBank

FS, WFS, N, GUARD, FRAME = 16000, 20000, 320, 80, 400

def pcm(x):
    return np.clip(np.rint(np.clip(np.asarray(x, dtype=np.float64), -1, 1)*32768), -32768, 32767).astype('<i2')

def floats(x):
    return np.asarray(x, dtype='<i2').astype(np.float32)/32768

def wav(path, rate):
    w = wave.open(str(path), 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    return w

def readn(f, n):
    buf = bytearray()
    while len(buf) < n:
        b = f.read(n-len(buf))
        if not b: break
        buf.extend(b)
    return bytes(buf)

def physical(kind, value):
    names = [r.split()[1] for r in pactl('list','short',kind).splitlines() if len(r.split())>1]
    if value:
        if value not in names: raise ValueError('Unknown device %s. Use: pactl list short %s' % (value, kind))
        return value
    label = 'Default Source:' if kind == 'sources' else 'Default Sink:'
    default = next((r.split(':',1)[1].strip() for r in pactl('info').splitlines() if r.startswith(label)), None)
    bad = {'tx_mic','rx_sink','tx_enc_sink'}
    if default in names and default not in bad and not default.endswith('.monitor'): return default
    matches = [x for x in names if x.startswith(('alsa_','bluez_')) and not x.endswith('.monitor')]
    if not matches: raise RuntimeError('No physical %s; select with --mic/--speaker' % kind)
    return matches[0]

def start_pcm_input(file_path, source_name, rate):
    if file_path:
        if not Path(file_path).is_file(): raise FileNotFoundError(file_path)
        if not shutil.which('ffmpeg'): raise RuntimeError('File input needs ffmpeg: sudo apt install ffmpeg')
        cmd = ['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(file_path),
               '-map','0:a:0','-vn','-ac','1','-ar',str(rate),'-f','s16le','-acodec','pcm_s16le','pipe:1']
    else:
        cmd = ['parec','--raw','--format=s16le','--channels=1','--rate='+str(rate),'--device='+source_name]
    return sp.Popen(cmd,stdout=sp.PIPE,bufsize=0)

def start_player(sink, rate):
    return sp.Popen(['paplay','--raw','--format=s16le','--channels=1','--rate='+str(rate),
                     '--device='+sink],stdin=sp.PIPE,bufsize=0)

def stop_proc(p, play=False):
    if p is None:return
    if play and p.stdin:
        try:p.stdin.close()
        except (OSError, BrokenPipeError):pass
        try:p.wait(timeout=3)
        except sp.TimeoutExpired:pass
    if p.poll() is None:p.terminate()
    try:p.wait(timeout=2)
    except sp.TimeoutExpired:p.kill();p.wait()

def mode_keys(stop, mode, capture, allow_mode):
    while not stop.is_set():
        try:ready,_,_=select.select([sys.stdin],[],[],0.2)
        except (OSError,ValueError):return
        if not ready:continue
        line=sys.stdin.readline()
        if not line:return
        s=line.strip().lower()
        if allow_mode and s in ('1','2','3'):
            mode[0]={'1':'raw','2':'correct','3':'wrong'}[s]
            print('[RX] MODE =',mode[0],flush=True)
        elif s in ('','q','stop'):
            stop.set()
            if capture and capture.poll() is None: capture.terminate()
            return

def make_sync(on_frame,on_event):
    return FixedGuardSyncV1(SyncV1Config(rate=WFS,frameSamples=FRAME,coreSamples=N,
        guardSamples=GUARD,spinnerSize=8,superSpinnerSize=8),on_frame,on_event)

def input_options(a):
    if a.scenario == 'message' and a.role == 'tx' and not a.source_file:
        raise ValueError('For voice-message TX supply --source-file /path/to/audio.wav')
    if a.source_file and a.role!='tx': raise ValueError('--source-file is for TX only')
    if a.received_file and a.role!='rx': raise ValueError('--received-file is for RX only')
    if a.scenario!='message' and a.received_file:raise ValueError('--received-file requires --scenario message')
    if a.guard_amp <= 0 or a.guard_amp > .5:raise ValueError('guard amp must be 0..0.5')

def tx(a):
    input_options(a)
    c=a.cipher(a.secret)
    mic=physical('sources',a.mic) if not a.source_file else None
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    bank=np.asarray(makeGuardBank(WFS,GUARD,64),dtype=np.float32)
    stop=threading.Event();capture=player=None;count=0;cpu=0.;source_done=False
    try:
        with VirtualAudio('tx'):
            print('\n[TX READY] Microphone: NearS_TX_Microphone (Pulse source: tx_mic)')
            print('Configure YOUR service to use NearS_TX_Microphone. No apps are started here.')
            if a.scenario == 'message':print('Open voice-message recording UI, press RECORD, then press ENTER here.')
            else:print('Join a call/start OBS streaming, then press ENTER here.')
            input('[TX] ENTER = START >> ')
            capture=start_pcm_input(a.source_file,mic,FS)
            player=start_player(TX_SINK,WFS)
            threading.Thread(target=mode_keys,args=(stop,None,capture,False),daemon=True).start()
            print('[TX] Sending; ENTER stops.',flush=True)
            started=time.monotonic()
            batch=int(getattr(c,'encode_batch_size',1))
            with wav(root/'tx_original_16k.wav',FS) as original, wav(root/'tx_wire_20k.wav',WFS) as sent:
                while not stop.is_set():
                    raw=readn(capture.stdout,N*2*batch)
                    if len(raw)<N*2*batch or stop.is_set():break
                    samples=floats(np.frombuffer(raw,dtype='<i2')).reshape(batch,N)
                    inds=np.arange(count,count+batch,dtype=np.int64)
                    t=time.perf_counter()
                    encrypted=(c.encode_batch(samples,inds) if batch>1 else np.asarray([c.encode(samples[0],count)]))
                    cpu+=time.perf_counter()-t
                    encrypted=np.asarray(encrypted,dtype=np.float32)
                    if encrypted.shape!=(batch,N):raise ValueError('Cipher payload must be (batch,320), got '+str(encrypted.shape))
                    wire=b''.join(pcm(np.r_[encrypted[j]*a.payload_gain,bank[int(index)%64]*a.guard_amp]).tobytes() for j,index in enumerate(inds))
                    original.writeframes(raw);sent.writeframes(wire);player.stdin.write(wire)
                    count+=batch
                    if count//250 != (count-batch)//250:
                        print('[TX] frames=%d seconds=%.1f avg_cipher_ms/frame=%.2f' % (count,count/50.,cpu*1000/max(1,count)),flush=True)
                    if a.duration and time.monotonic()-started >= a.duration:break
            if a.source_file and player and not stop.is_set():
                # Wait until paplay actually finishes transmitting its buffered samples.
                # Without draining, an instant FFmpeg file read could end recording early.
                try:
                    player.stdin.close()
                    player.wait()
                except (OSError, BrokenPipeError):pass
                player=None
            source_done=not stop.is_set()
            if source_done:print('[TX] Finished playback. Close/send the voice message in the service.',flush=True)
            if a.scenario=='message' and source_done:
                print('[TX] Virtual microphone stays alive until ENTER, so you can finish recording.')
                # Input was already consumed by the keyboard thread; avoid two stdin readers.
                stop.wait()  # user stops with ENTER in terminal; keeps virtual mic alive
    except (KeyboardInterrupt,BrokenPipeError):pass
    finally:
        stop.set();stop_proc(capture);stop_proc(player,True);c.close()
        summary=dict(scenario=a.scenario,frames=count,source_completed=source_done,encode_cpu_sec=cpu,
                     source_file=a.source_file,wire_rate=WFS)
        (root/'summary.json').write_text(json.dumps(summary,indent=2))
        print('[TX] Output:',root,flush=True)

def rx(a):
    input_options(a)
    c=a.cipher(a.secret);root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=True)
    speaker=physical('sinks',a.speaker) if a.play else None
    stop=threading.Event();mode=[a.listen_mode];capture=player=None;sync=None
    stats=dict(scenario=a.scenario,ok=0,lost=0,decoded=0,decode_errors=0,decode_cpu_sec=0.,rx_samples=0,selected_mode=a.listen_mode)
    try:
        # --received-file: decode a downloaded file directly. No virtual sink is needed.
        from contextlib import nullcontext
        with (nullcontext() if a.received_file else VirtualAudio('rx')):
            if a.received_file:
                print('[RX] Input from received media file:',a.received_file)
            else:
                print('\n[RX READY] Virtual speakers: NearS_RX (Pulse sink: rx_sink)')
                print('Set YOUR app/browser playback OUTPUT to NearS_RX.')
                if a.scenario=='message':print('Play received voice message in the service AFTER pressing ENTER here.')
                else:print('Join the call/watch the stream; start RX before TX if possible.')
            input('[RX] ENTER = START >> ')
            capture=start_pcm_input(a.received_file,RX_SINK+'.monitor',WFS)
            if a.play:player=start_player(speaker,FS)
            threading.Thread(target=mode_keys,args=(stop,mode,capture,True),daemon=True).start()
            print('[RX] 1+ENTER RAW, 2+ENTER CORRECT, 3+ENTER WRONG, empty ENTER STOP',flush=True)
            begun=time.monotonic()
            with wav(root/'rx_incoming_20k.wav',WFS) as incoming, wav(root/'rx_cipher_16k.wav',FS) as cipher_wav, \
                 wav(root/'rx_listened_16k.wav',FS) as listened, (root/'frames.csv').open('w',newline='') as csv_f, \
                 (root/'events.jsonl').open('w') as events:
                writer=csv.writer(csv_f);writer.writerow(['frame_index','status','corr','mode','decode_ms'])
                def frame(index,packet,status,meta):
                    index=int(index)+a.index_offset
                    valid=status=='ok';stats['ok' if valid else 'lost']+=1
                    wire=np.clip(np.asarray(packet,dtype=np.float32)[:N]/a.payload_gain,-1,1) if valid else np.zeros(N,dtype=np.float32)
                    cipher_wav.writeframes(pcm(wire).tobytes())
                    current=mode[0];t=time.perf_counter()
                    try:
                        if current=='raw':decoded=wire
                        elif not valid:decoded=np.zeros(N,dtype=np.float32)
                        else:
                            target=index+(a.wrong_shift if current=='wrong' else 0)
                            decoded=(c.decode_frame(wire,target) if hasattr(c,'decode_frame') else c.decode(wire,target))
                            stats['decoded']+=1
                        decoded=np.asarray(decoded,dtype=np.float32).reshape(N)
                    except Exception as e:
                        stats['decode_errors']+=1;decoded=np.zeros(N,np.float32)
                        if stats['decode_errors']<=5:print('[RX] DECODE ERROR',index,repr(e),flush=True)
                    dt=time.perf_counter()-t;stats['decode_cpu_sec']+=dt
                    b=pcm(decoded).tobytes();listened.writeframes(b)
                    if player:
                        try:player.stdin.write(b)
                        except BrokenPipeError:stop.set()
                    writer.writerow([index,status,meta.get('corr',''),current,round(dt*1000,3)])
                    if (stats['ok']+stats['lost'])%250==0:print('[RX] ok=%d lost=%d decoded=%d mode=%s'%(stats['ok'],stats['lost'],stats['decoded'],current),flush=True)
                def on_event(ev):
                    if hasattr(c,'on_sync_event'):c.on_sync_event(ev)
                    events.write(json.dumps(ev,default=str)+'\n')
                sync=make_sync(frame,on_event)
                # 173 samples (odd to 400), no reliance on source packet boundaries.
                tail=np.empty(0,np.float32)
                while not stop.is_set():
                    b=readn(capture.stdout,2*173)
                    if len(b)<2*173:
                        if b and len(b)%2==0: sync.push(floats(np.frombuffer(b,dtype='<i2')),time.monotonic());stats['rx_samples']+=len(b)//2
                        break
                    incoming.writeframes(b);x=floats(np.frombuffer(b,dtype='<i2'))
                    sync.push(x,time.monotonic());stats['rx_samples']+=len(x)
                    if a.duration and time.monotonic()-begun>=a.duration:break
                # Explicit flush, including after EOF for media files.
                if sync and not stop.is_set():sync.finish()
                stats['selected_mode']=mode[0]
    except (KeyboardInterrupt,BrokenPipeError):pass
    finally:
        stop.set();stop_proc(capture);stop_proc(player,True);c.close()
        if sync:stats.update(sync_stats=sync.stats,final_state=sync.state,last_frame=sync.lastFrame)
        (root/'summary.json').write_text(json.dumps(stats,indent=2,default=str))
        print('[RX] Output:',root,flush=True)

def selftest(a):
    c=a.cipher(a.secret);bank=np.asarray(makeGuardBank(WFS,GUARD,64),np.float32)
    source=np.random.default_rng(1818).normal(0,.07,(64,N)).astype(np.float32)
    received=[];s=make_sync(lambda i,p,st,meta:received.append((int(i),st,np.asarray(p,dtype=np.float32)[:N].copy())),lambda e:None)
    batch=int(getattr(c,'encode_batch_size',1));data=[]
    for i in range(0,len(source),batch):
        x=source[i:i+batch];ids=np.arange(i,i+len(x),dtype=np.int64)
        encoded=(c.encode_batch(x,ids) if batch>1 else np.asarray([c.encode(x[0],i)]))
        for j,item in enumerate(encoded):data.append(pcm(np.r_[item*a.payload_gain,bank[(i+j)%64]*a.guard_amp]))
    wire=floats(np.concatenate(data))
    for i in range(0,len(wire),173):s.push(wire[i:i+173],time.monotonic())
    s.finish();assert len(received)==64 and all(i==j and status=='ok' for j,(i,status,_) in enumerate(received))
    errs=[]
    for i,_,p in received:
        data=p/a.payload_gain
        x=(c.decode_frame(data,i) if hasattr(c,'decode_frame') else c.decode(data,i))
        errs.append(np.mean((np.asarray(x)-source[i])**2))
    print('[PASS] %s 64/64 synced, RMSE %.5f'%(a.name,np.sqrt(np.mean(errs))),flush=True)
    c.close()

def main(cipher,name,default_scenario='call'):
    p=argparse.ArgumentParser(description='NearS '+name+' service-neutral audio test')
    p.add_argument('--role',required=True,choices=['tx','rx','selftest'])
    p.add_argument('--scenario',choices=['call','message'],default=default_scenario)
    p.add_argument('--source-file',help='TX: any ffmpeg-decodable audio file, mono/resampled to 16k internally')
    p.add_argument('--received-file',help='RX message: decoded audio/media downloaded from service, internally 20k mono')
    p.add_argument('--secret',default=os.getenv('NEARS_ALGO_SECRET','nears-test'))
    p.add_argument('--listen-mode',choices=['raw','correct','wrong'],default='correct')
    p.add_argument('--wrong-shift',type=int,default=64)
    p.add_argument('--index-offset',type=int,default=0,help='RX: absolute frame offset; use for delayed streams if known')
    p.add_argument('--payload-gain',type=float,default=1.)
    p.add_argument('--guard-amp',type=float,default=.12)
    p.add_argument('--mic');p.add_argument('--speaker')
    p.add_argument('--no-play',action='store_false',dest='play',help='RX do not play into physical speakers')
    p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--out',default='/tmp/nears_service_'+name)
    a=p.parse_args();a.cipher=cipher;a.name=name
    if a.index_offset%64: p.error('--index-offset must be a multiple of 64; PN cycle=64')
    if not (0<a.payload_gain<=1):p.error('--payload-gain must be in (0,1]')
    if a.role=='tx':tx(a)
    elif a.role=='rx':rx(a)
    else:selftest(a)
