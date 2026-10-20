"""TimePack 20k STFT/MDCT adapter, reusing existing SuperSpinner.

Original streaming STFT/MDCT code lives under spectral_core/crypto.
No legacy synchronization or virtual-device code from git_core is installed.

Original bridge: Two-way TX/RX bridge for TimePack 20k with pluggable algorithm-owned cipher.

Run from project root: python3 -m experiment.spectralTimepack20k --role tx|rx ...
Uses experiment.telemostTest for auto devices / Chrome routing; does not modify it.
"""
import argparse
import contextlib
import importlib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

# Limit BLAS oversubscription BEFORE numpy import, unlike after-the-fact changes.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.signal import resample_poly
from crypto.frame_port import FrameContext, FrameKeySource, IdentityPlugin, FRAME, check_batch
from experiment.telemostTest import AutoRouter, VirtualAudio, pactl, TX_SINK, RX_SINK

RATE, OUT_RATE, CYCLE_SAMPLES, GUARD_SAMPLES = 16000,20000,400,80


def _pcm(x):
    return np.rint(np.clip(x,-1,1)*32767.0).astype('<i2').tobytes()


def _wav(path, rate):
    f = wave.open(str(path),'wb')
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(rate)
    return f


def _get_cipher(args):
    if args.algo == 'spectral':
        if not args.key_file:
            raise RuntimeError('--algo spectral needs --key-file PATH (raw bytes, identical TX/RX)')
        from crypto.spectral_timepack_adaptor import SpectralTimepackAdaptor
        cipher = SpectralTimepackAdaptor(
            key_file=args.key_file, transform=args.transform, operation=args.operation,
            key_schedule=args.key_schedule, session=args.session,
            stft_frame=args.stft_frame, stft_hop=args.stft_hop,
            mdct_half=args.mdct_half, max_gap_frames=args.max_gap_frames)
        print('SPECTRAL: %s/%s schedule=%s spectral_frame=%s hop=%s delay=%d transport frames (one direction)' % (
            args.transform, args.operation, args.key_schedule, cipher.frame_samples,
            cipher.hop_samples, cipher.delay_frames), flush=True)
        return cipher
    if args.algo == 'identity':
        return IdentityPlugin()
    if args.algo == 'alg7_kra':
        if not args.algo_dir:
            raise RuntimeError('--algo-dir must point to the unmodified original ALG7 files')
        if args.key_file:
            raise RuntimeError('ALG7 original RNG does not use --key-file; use --alg7-seed matching both sides')
        if args.alg7_seed is not None and args.demo_seed is not None:
            raise RuntimeError('Choose --alg7-seed OR --demo-seed, not both')
        seed = args.alg7_seed if args.alg7_seed is not None else (args.demo_seed if args.demo_seed is not None else 37)
        from crypto.alg7_kra_adaptor import Alg7KraAdaptor
        print('ALG7: ORIGINAL np.random.default_rng(seed=%d), original Encryption_class unchanged; this seed must match TX/RX' % seed, flush=True)
        return Alg7KraAdaptor(args.algo_dir, seed=seed, output_gain=args.output_gain)
    if args.algo == 'pseudo':
        raise RuntimeError('Old --algo pseudo used HMAC instead of original RNG. Use --algo alg7_kra to preserve the original algorithm.')
    if args.algo != 'plugin':
        raise ValueError('Unknown cipher: %s' % args.algo)
    if not args.plugin or ':' not in args.plugin:
        raise ValueError('--algo plugin requires --plugin package.module:ClassName')
    # Generic plugins configure themselves. The bridge never decides which
    # cipher/key policy a plugin has to use. Optional legacy HMAC remains
    # available ONLY for plugins that explicitly accept a `keys` parameter.
    module_name, class_name = args.plugin.rsplit(':', 1)
    plugin_cls = getattr(importlib.import_module(module_name), class_name)
    import inspect
    params = inspect.signature(plugin_cls).parameters
    kwargs = {}
    if 'algo_dir' in params: kwargs['algo_dir'] = args.algo_dir
    if 'output_gain' in params: kwargs['output_gain'] = args.output_gain
    if 'keys' in params:
        if args.key_file:
            kwargs['keys'] = FrameKeySource.from_file(args.key_file, args.session)
        elif args.demo_seed is not None:
            kwargs['keys'] = FrameKeySource.for_demo_only(args.demo_seed, args.session)
        else:
            raise RuntimeError('This plugin requires `keys`; provide --key-file or --demo-seed, or implement your own key source in the plugin')
    return plugin_cls(**kwargs)


def _guards():
    from sync.v2_timepack import makeGuardBank
    bank = makeGuardBank(OUT_RATE,GUARD_SAMPLES,64)
    if np.asarray(bank).shape!=(64,GUARD_SAMPLES):
        raise RuntimeError('v2_timepack makeGuardBank must return (64,80) PN templates')
    return bank


def _frame_to_wire(clear_cipher_frame,idx,guardbank,gain,guard_amp):
    frame = np.asarray(clear_cipher_frame,dtype=np.float32)
    if frame.shape!=(FRAME,):
        raise RuntimeError('plugin.encode_batch must emit 320 samples / logical frame')
    out = np.empty(CYCLE_SAMPLES,dtype=np.float32)
    out[:320]=frame*gain
    out[320:]=guardbank[idx%64]*guard_amp
    return out


def _iter_wav_batches(path, batch_size=4, limit=0.0):
    with wave.open(str(path),'rb') as w:
        if w.getnchannels()!=1 or w.getsampwidth()!=2:
            raise ValueError('TX input must be PCM int16 mono WAV')
        in_rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float64)
    if in_rate != RATE:
        from math import gcd
        g = gcd(in_rate,RATE)
        data = resample_poly(data,RATE//g,in_rate//g)
    if limit:
        data = data[:int(limit*RATE)]
    step=batch_size*FRAME
    for off in range(0,len(data),step):
        part=data[off:off+step]
        if len(part)<step:
            part=np.pad(part,(0,step-len(part)))
        yield (part.astype(np.float32)/32768.0).reshape(batch_size,FRAME)


def _read_exact(stream, nbytes):
    parts=[]
    while nbytes>0:
        b=stream.read(nbytes)
        if not b:
            break
        parts.append(b)
        nbytes-=len(b)
    return b''.join(parts)


def _live_batches(batch_size=4):
    default_source = next((line.split(':',1)[1].strip() for line in pactl('info').splitlines() if line.startswith('Default Source:')),None)
    if not default_source or default_source=='tx_mic':
        raise RuntimeError('Physical Default Source not found; select physical mic as PulseAudio default source')
    cmd=['parec',f'--device={default_source}','--raw','--format=s16le',f'--rate={RATE}','--channels=1','--latency-msec=30']
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,bufsize=0)
    try:
        while True:
            b=_read_exact(p.stdout,batch_size*FRAME*2)
            if len(b)<batch_size*FRAME*2:
                break
            yield np.frombuffer(b,dtype='<i2').astype(np.float32).reshape(batch_size,FRAME)/32768.0
    finally:
        p.terminate()
        try:p.wait(timeout=2)
        except subprocess.TimeoutExpired:p.kill();p.wait()


def _generate_frames(cipher, input_batches, guardbank, outfile, duration=0.0, player=None, gain=0.75, guard_amp=0.12):
    start=time.perf_counter()
    stats={'tx_frames':0,'cipher_cpu_sec':0.0,'wire_rate':OUT_RATE,'payload_samples':FRAME,'guard_samples':GUARD_SAMPLES}
    max_frames = int(round(duration*50)) if duration else None
    for batch in input_batches:
        batch=check_batch(batch,np.arange(len(batch)))
        first=stats['tx_frames']
        if max_frames is not None and first>=max_frames:
            break
        indices=np.arange(first,first+len(batch),dtype=np.int64)
        t=time.process_time()
        encoded=cipher.encode_batch(batch,indices)
        stats['cipher_cpu_sec']+=time.process_time()-t
        encoded=check_batch(encoded,indices)
        count=len(encoded) if max_frames is None else min(len(encoded),max_frames-first)
        pcm=b''.join(_pcm(_frame_to_wire(encoded[j],int(indices[j]),guardbank,gain,guard_amp)) for j in range(count))
        outfile.writeframes(pcm)
        if player is not None:
            player.stdin.write(pcm)
        stats['tx_frames']+=count
        if first%250==0:
            print('TX crypto frames:',stats['tx_frames'],'cipher_cpu_sec:',round(stats['cipher_cpu_sec'],2),flush=True)
    stats['elapsed_wall_sec']=time.perf_counter()-start
    return stats


def _handle_tx(args,out):
    plugin=_get_cipher(args)
    guardbank=_guards()
    out.mkdir(parents=True,exist_ok=True)
    wavfile=out/'tx_encrypted_20k.wav'
    stats={}
    try:
        if args.input:
            print('Preparing TX encrypted 20k WAV, without virtual devices...',flush=True)
            with _wav(wavfile,OUT_RATE) as wf:
                stats=_generate_frames(plugin,_iter_wav_batches(args.input,plugin.encode_batch_size,args.duration),guardbank,wf,0,None,args.tx_gain,args.guard_amp)
            print('TX encrypted:',wavfile,'frames:',stats['tx_frames'])
        if args.generate_only:
            if not args.input:
                raise ValueError('--generate-only needs --input WAV')
            return
        with VirtualAudio('tx'):
            print("Select NearS_TX_Microphone as Telemost microphone; wait for RX, then ENTER.",flush=True)
            input('PRESS ENTER to start encrypted TX... ')
            with AutoRouter('tx') as router:
                router.ensure_tx()
                if args.input:
                    subprocess.run(['paplay',f'--device={TX_SINK}',str(wavfile)],check=True)
                else:
                    max_seconds=args.duration or 30.0
                    print('Capturing physical microphone; encoding live in batches of 4 frames.',flush=True)
                    cmd=['paplay','--raw',f'--device={TX_SINK}',f'--rate={OUT_RATE}','--format=s16le','--channels=1']
                    player=subprocess.Popen(cmd,stdin=subprocess.PIPE)
                    try:
                        with _wav(wavfile,OUT_RATE) as wf:
                            stats=_generate_frames(plugin,_live_batches(plugin.encode_batch_size),guardbank,wf,max_seconds,player,args.tx_gain,args.guard_amp)
                    finally:
                        if player.stdin:player.stdin.close()
                        player.wait(timeout=5)
            print('TX ended. Stop RX after a couple of seconds.',flush=True)
    finally:
        plugin.close()
        if stats:
            (out/'tx_summary.json').write_text(json.dumps(stats,indent=2)+'\n')


def _rx_source_pcm(args):
    if args.wav:
        with wave.open(args.wav,'rb') as w:
            if w.getnchannels()!=1 or w.getframerate()!=OUT_RATE or w.getsampwidth()!=2:
                raise ValueError('RX --wav must be mono 20kHz s16 WAV')
            while True:
                block=w.readframes(args.io_samples)
                if not block:break
                yield block
        return
    proc=subprocess.Popen(['parec',f'--device={RX_SINK}.monitor','--raw','--format=s16le',f'--rate={OUT_RATE}','--channels=1','--latency-msec=10'],stdout=subprocess.PIPE,bufsize=0)
    try:
        while True:
            block=_read_exact(proc.stdout,args.io_samples*2)
            if not block:break
            yield block
    finally:
        proc.terminate()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


def _handle_rx(args,out):
    from sync.v2_timepack import FixedGuardSyncV1,SyncV1Config
    cipher=_get_cipher(args)
    out.mkdir(parents=True,exist_ok=True)
    frames_csv=open(out/'frames.csv','w',newline='')
    events_json=open(out/'events.jsonl','w')
    csvwriter=csv.writer(frames_csv)
    csvwriter.writerow(['frame','status','guard_pos','corr','reason','decrypt_ms'])
    summaries={'decrypted_frames':0,'skipped_lost_frames':0,'decrypt_cpu_sec':0.0,'input_audio_sec':0.0,'push_cpu_sec':0.0,'push_wall_sec':0.0,'max_push_ms':0.0}
    if hasattr(cipher, 'delay_frames'):
        summaries.update(spectral_transform=getattr(cipher,'transform',''), spectral_operation=getattr(cipher,'operation',''), spectral_hop_samples=cipher.hop_samples, startup_delay_transport_frames_each_direction=cipher.delay_frames)
    default_sink=next((line.split(':',1)[1].strip() for line in pactl('info').splitlines() if line.startswith('Default Sink:')),None) if args.listen else None
    speaker=None
    try:
        with contextlib.ExitStack() as stack:
            clear_w=stack.enter_context(contextlib.closing(_wav(out/'rx_decrypted_16k.wav',RATE)))
            cipher_w=stack.enter_context(contextlib.closing(_wav(out/'rx_cipher_payload_16k.wav',RATE)))
            raw_w=stack.enter_context(contextlib.closing(_wav(out/'rx_raw_20k.wav',OUT_RATE)))
            if args.listen and not args.wav:
                if not default_sink: raise RuntimeError('Cannot find physical speakers for --listen')
                cmd=['paplay','--raw',f'--device={default_sink}',f'--rate={RATE}','--format=s16le','--channels=1']
                speaker=subprocess.Popen(cmd,stdin=subprocess.PIPE)
            def on_frame(index,wireframe,status,meta):
                core=np.asarray(wireframe,dtype=np.float32)[:FRAME] if status=='ok' else None
                if core is not None:
                    # Undo fixed TX gain for legacy feature extractor.
                    core=np.clip(core/args.tx_gain,-1,1)
                    cipher_w.writeframes(_pcm(core))
                else:
                    cipher_w.writeframes(bytes(FRAME*2))
                t=time.process_time()
                decoded=cipher.decode_frame(core,FrameContext(int(index),status))
                seconds=time.process_time()-t
                decoded=np.asarray(decoded,dtype=np.float32)
                if decoded.shape!=(FRAME,):
                    raise ValueError('plugin.decode_frame returned wrong shape')
                output=_pcm(decoded)
                clear_w.writeframes(output)
                if speaker and speaker.stdin:
                    speaker.stdin.write(output)
                summaries['decrypt_cpu_sec']+=seconds
                summaries['decrypted_frames']+=1
                if status != 'ok':summaries['skipped_lost_frames']+=1
                csvwriter.writerow((index,status,meta.get('guard_pos',''),meta.get('corr',''),meta.get('reason',''),round(seconds*1000,5)))
            def on_event(event):
                cipher.on_sync_event(event)
                events_json.write(json.dumps(event)+'\n')
                events_json.flush()
                if event.get('event') in ('acquired','reacquired','track_miss'):
                    print('SYNC',event.get('event'),event,flush=True)
            cfg=SyncV1Config(rate=OUT_RATE,frameSamples=CYCLE_SAMPLES,coreSamples=FRAME,guardSamples=GUARD_SAMPLES,spinnerSize=8,superSpinnerSize=8)
            sync=FixedGuardSyncV1(cfg,on_frame,on_event)
            started=time.monotonic()
            try:
                for block in _rx_source_pcm(args):
                    if args.duration and time.monotonic()-started>=args.duration:
                        break
                    raw_w.writeframes(block)
                    pcm=np.frombuffer(block,dtype='<i2').astype(np.float32)/32768.0
                    summaries['input_audio_sec']+=len(pcm)/OUT_RATE
                    t=time.perf_counter()
                    cpu=time.process_time()
                    sync.push(pcm,time.monotonic())
                    summaries['push_wall_sec']+=time.perf_counter()-t
                    summaries['push_cpu_sec']+=time.process_time()-cpu
                    summaries['max_push_ms']=max(summaries['max_push_ms'],(time.perf_counter()-t)*1000)
                    if int(summaries['input_audio_sec'])%10==0 and int(summaries['input_audio_sec']-len(pcm)/OUT_RATE)%10!=0:
                        print('PERF audio_sec=%.1f sync_cpu_ratio=%.3f crypto_cpu_ratio=%.3f' % (summaries['input_audio_sec'],summaries['push_cpu_sec']/max(1e-6,summaries['input_audio_sec']),summaries['decrypt_cpu_sec']/max(1e-6,summaries['input_audio_sec'])),flush=True)
                if args.wav:sync.finish()
            except KeyboardInterrupt:
                print('RX stopped by Ctrl+C',flush=True)
            summaries.update(sync.stats)
            summaries['state_final']=sync.state
            summaries['last_frame']=sync.lastFrame
            # sync.push() invokes on_frame() synchronously, hence push CPU includes
            # plugin decode CPU. Subtract to avoid double-counting cipher work.
            audio=max(1e-9,summaries['input_audio_sec'])
            summaries['push_total_cpu_ratio']=summaries['push_cpu_sec']/audio
            summaries['sync_cpu_ratio']=max(0.0,summaries['push_cpu_sec']-summaries['decrypt_cpu_sec'])/audio
            summaries['decrypt_cpu_ratio']=summaries['decrypt_cpu_sec']/audio
            if hasattr(cipher, 'lost_input_frames'):
                summaries['cipher_lost_input_frames']=cipher.lost_input_frames
                summaries['cipher_implicit_gap_frames']=cipher.implicit_gap_frames
    finally:
        if speaker:
            try:speaker.stdin.close()
            except Exception:pass
            try:speaker.wait(timeout=3)
            except subprocess.TimeoutExpired:speaker.terminate();speaker.wait()
        cipher.close()
        frames_csv.close()
        events_json.close()
        (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print('RX summary:',json.dumps(summaries,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description='NearS TimePack 20k: independent TX/RX cipher plugin bridge')
    p.add_argument('--role',choices=['tx','rx'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--algo',choices=['spectral','identity','alg7_kra','pseudo','plugin'],default='spectral')
    p.add_argument('--transform',choices=['stft','mdct'],default='stft')
    p.add_argument('--operation',choices=['invert','shuffle','signflip'],default='signflip')
    p.add_argument('--key-schedule',choices=['fixed','per_frame'],default='per_frame')
    p.add_argument('--stft-frame',type=int,default=2048)
    p.add_argument('--stft-hop',type=int,default=1024)
    p.add_argument('--mdct-half',type=int,default=2500)
    p.add_argument('--max-gap-frames',type=int,default=4096)
    p.add_argument('--plugin',help='For --algo plugin: import.path:ClassName')
    p.add_argument('--algo-dir',help='Directory containing unmodified user algorithm and models')
    p.add_argument('--key-file',help='hex master key in file; identical on both sides')
    p.add_argument('--demo-seed',type=int,help='Legacy generic plugin demo seed; also accepted as ALG7 original RNG seed')
    p.add_argument('--alg7-seed',type=int,help='Original ALG7 np.random.default_rng(seed), default=37. Not secure as a public value.')
    p.add_argument('--session',default='stft-live-v1',help='Same session string on both sides')
    p.add_argument('--input',help='TX: mono WAV speech input, pre-encrypted before Telemost')
    p.add_argument('--generate-only',action='store_true',help='TX: prepare WAV only, without creating devices')
    p.add_argument('--wav',help='RX: offline captured TimePack 20k WAV instead of Telemost')
    p.add_argument('--duration',type=float,default=0.0,help='TX live duration; TX input clip; RX live maximum')
    p.add_argument('--listen',action='store_true',help='RX: play DECRYPTED speech on physical speakers')
    p.add_argument('--tx-gain',type=float,default=.75)
    p.add_argument('--guard-amp',type=float,default=.12)
    p.add_argument('--output-gain',type=float,default=1.0)
    p.add_argument('--io-samples',type=int,default=200)
    args=p.parse_args()
    if args.io_samples<=0 or args.tx_gain<=0 or args.guard_amp<=0:
        p.error('positive gains and io-samples required')
    out=Path(args.out).expanduser().resolve()
    if args.role=='tx':
        _handle_tx(args,out)
    elif args.wav:
        _handle_rx(args,out)
    else:
        with VirtualAudio('rx'):
            print('TELEMOST RX: select SPEAKERS NearS_RX, then ENTER.',flush=True)
            input('PRESS ENTER to record/decrypt ... ')
            with AutoRouter('rx'):
                _handle_rx(args,out)

if __name__=='__main__':
    main()
