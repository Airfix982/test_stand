"""Live microphone / Telemost proof with three simultaneous RX audio branches.

Uses unchanged existing sync/v2_timepack.py and experiment/telemostTest.py.
Do not use VoiceProofAdaptor for confidential communications.
"""
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import subprocess
import time
import wave

for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v]='1'
import numpy as np

from crypto.frame_port import FRAME, FrameContext, check_batch
from crypto.voiceproof_adaptor import VoiceProofAdaptor
from experiment.spectralTimepack20k import (_guards, _frame_to_wire, _pcm, _wav, _live_batches,
    _iter_wav_batches, _physical_endpoint, _rx_source_pcm, AutoRouter, VirtualAudio,
    RATE, OUT_RATE, TX_SINK)


def make_cipher(args, offset=0):
    return VoiceProofAdaptor(args.key_file,session=args.session,grain=args.grain,index_offset=offset)


def tx(args, out):
    cipher=make_cipher(args)
    bank=_guards()
    out.mkdir(parents=True,exist_ok=True)
    mic=_physical_endpoint('sources',args.mic_source) if not args.input else None
    if mic: print('PHYSICAL MIC:',mic,flush=True)
    stats={'role':'tx','tx_frames':0,'cipher_cpu_sec':0.0,'transport_rate':OUT_RATE,'plaintext_rate':RATE,'grain':args.grain}

    def process(player=None):
        batches = _iter_wav_batches(args.input, cipher.encode_batch_size, args.duration) if args.input else _live_batches(cipher.encode_batch_size,mic)
        max_frames=int(round(args.duration*50)) if args.duration>0 else None
        with contextlib.ExitStack() as stack:
            plain=stack.enter_context(contextlib.closing(_wav(out/'tx_plain_16k.wav',RATE)))
            wire=stack.enter_context(contextlib.closing(_wav(out/'tx_encrypted_20k.wav',OUT_RATE))) if args.record_wire or args.generate_only else None
            try:
                for batch in batches:
                    start_index=stats['tx_frames']
                    if max_frames is not None and start_index>=max_frames: break
                    count=min(len(batch),max_frames-start_index) if max_frames is not None else len(batch)
                    indices=np.arange(start_index,start_index+count,dtype=np.int64)
                    batch=check_batch(batch[:count],indices)
                    t=time.process_time()
                    encrypted=cipher.encode_batch(batch,indices)
                    stats['cipher_cpu_sec']+=time.process_time()-t
                    encrypted=check_batch(encrypted,indices)
                    data=b''.join(_pcm(_frame_to_wire(encrypted[i],int(n),bank,args.tx_gain,args.guard_amp)) for i,n in enumerate(indices))
                    plain.writeframes(_pcm(batch.ravel()))
                    if wire:wire.writeframes(data)
                    if player:player.stdin.write(data)
                    stats['tx_frames']+=count
                    if stats['tx_frames']%250< count:
                        audio=stats['tx_frames']/50.
                        rms=float(np.sqrt(np.mean(batch.astype(np.float64)**2)))
                        print('TX %.1fs / cpu_ratio=%.3f / mic_rms=%.1fdBFS' % (audio,stats['cipher_cpu_sec']/audio,20*np.log10(max(rms,1e-9))),flush=True)
            except KeyboardInterrupt:
                print('TX: stopped by Ctrl+C',flush=True)
            finally:
                close=getattr(batches,'close',None)
                if callable(close):close()
    try:
        if args.generate_only:
            if not args.input:raise ValueError('--generate-only requires --input WAV')
            print('OFFLINE TX: no devices',flush=True)
            process()
        else:
            with VirtualAudio('tx'):
                print('Telemost: MICROPHONE = NearS_TX_Microphone. RX should already be recording.',flush=True)
                input('PRESS ENTER to start sending LIVE SCRAMBLED AUDIO... ')
                with AutoRouter('tx') as router:
                    router.ensure_tx()
                    player=subprocess.Popen(['paplay','--raw',f'--device={TX_SINK}',f'--rate={OUT_RATE}','--format=s16le','--channels=1'],stdin=subprocess.PIPE,bufsize=0)
                    try:process(player)
                    finally:
                        if player.stdin:
                            try:player.stdin.close()
                            except BrokenPipeError:pass
                        try:player.wait(timeout=4)
                        except subprocess.TimeoutExpired:player.terminate();player.wait()
    finally:
        cipher.close()
        stats['cipher_cpu_ratio']=stats['cipher_cpu_sec']/max(1e-9,stats['tx_frames']/50.)
        (out/'tx_summary.json').write_text(json.dumps(stats,indent=2)+'\n')
        print('TX DONE:',json.dumps(stats,indent=2),flush=True)


def rx(args,out):
    from sync.v2_timepack import FixedGuardSyncV1,SyncV1Config
    out.mkdir(parents=True,exist_ok=True)
    correct=make_cipher(args)
    wrong=make_cipher(args,offset=args.wrong_offset)
    stats={'role':'rx','frames_out':0,'lost':0,'audio_in_sec':0.0,'cpu_total_sec':0.0,'cpu_correct_wrong_sec':0.0,
           'correct_absolute_offset':0,'wrong_decoding_absolute_offset':args.wrong_offset,'grain':args.grain}
    srcsink=_physical_endpoint('sinks',args.speaker_sink) if args.listen and not args.wav else None
    if srcsink:print('PHYSICAL HEADPHONES:',srcsink,flush=True)
    csvf=open(out/'frames.csv','w',newline='')
    evf=open(out/'events.jsonl','w')
    writer=csv.writer(csvf)
    writer.writerow(['frame','status','guard_pos','corr','reason'])
    player=None
    try:
        with contextlib.ExitStack() as stack:
            allw={
                'cipher':stack.enter_context(contextlib.closing(_wav(out/'rx_cipher_payload_16k.wav',RATE))),
                'correct':stack.enter_context(contextlib.closing(_wav(out/'rx_decrypted_16k.wav',RATE))),
                'wrong':stack.enter_context(contextlib.closing(_wav(out/'rx_wrong_index_16k.wav',RATE))),
            }
            raw=stack.enter_context(contextlib.closing(_wav(out/'rx_raw_20k.wav',OUT_RATE)))
            if args.listen and not args.wav:
                player=subprocess.Popen(['paplay','--raw',f'--device={srcsink}',f'--rate={RATE}','--format=s16le','--channels=1'],stdin=subprocess.PIPE,bufsize=0)
            previous_monitor=None

            def on_frame(index,wireframe,status,meta):
                nonlocal previous_monitor
                frame=np.asarray(wireframe,dtype=np.float32)[:FRAME] if status=='ok' else None
                if frame is not None:frame=np.clip(frame/args.tx_gain,-1,1)
                t=time.process_time()
                outputs={
                    'cipher':frame if frame is not None else np.zeros(FRAME,np.float32),
                    'correct':correct.decode_frame(frame,FrameContext(int(index),status)),
                    'wrong':wrong.decode_frame(frame,FrameContext(int(index),status)),
                }
                stats['cpu_correct_wrong_sec']+=time.process_time()-t
                for k,v in outputs.items(): allw[k].writeframes(_pcm(v))
                if player and player.stdin:
                    choice=args.monitor
                    if choice=='rotate':choice=('cipher','correct','wrong')[int(index)//250%3]
                    if choice !=previous_monitor:
                        previous_monitor=choice
                        print('HEADPHONES MONITOR:',choice, 'frame', index,flush=True)
                    player.stdin.write(_pcm(outputs[choice]))
                writer.writerow([index,status,meta.get('guard_pos',''),meta.get('corr',''),meta.get('reason','')])
                stats['frames_out']+=1
                if status!='ok':stats['lost']+=1

            def on_event(event):
                correct.on_sync_event(event)
                wrong.on_sync_event(event)
                evf.write(json.dumps(event)+'\n')
                evf.flush()
                if event.get('event') in ('acquired','reacquired','track_miss'):
                    print('SYNC:',event,flush=True)

            cfg=SyncV1Config(rate=OUT_RATE,frameSamples=400,coreSamples=320,guardSamples=80,spinnerSize=8,superSpinnerSize=8)
            sync=FixedGuardSyncV1(cfg,on_frame,on_event)
            def capture():
                started=time.monotonic()
                try:
                    for block in _rx_source_pcm(args):
                        if args.duration and time.monotonic()-started>=args.duration:break
                        raw.writeframes(block)
                        pcm=np.frombuffer(block,dtype='<i2').astype(np.float32)/32768.0
                        stats['audio_in_sec']+=len(pcm)/OUT_RATE
                        t=time.process_time()
                        sync.push(pcm,time.monotonic())
                        stats['cpu_total_sec']+=time.process_time()-t
                        if int(stats['audio_in_sec'])%10==0 and int(stats['audio_in_sec']-len(pcm)/OUT_RATE)%10!=0:
                            print('RX %.1fs, frames=%d, lost=%d' % (stats['audio_in_sec'],stats['frames_out'],stats['lost']),flush=True)
                    if args.wav:sync.finish()
                except KeyboardInterrupt:
                    print('RX: stopped by Ctrl+C',flush=True)
            if args.wav: capture()
            else:
                with VirtualAudio('rx'):
                    print('Telemost: SPEAKERS = NearS_RX (NOT physical headphones)',flush=True)
                    input('PRESS ENTER to start RX / capture... ')
                    with AutoRouter('rx'):capture()
            stats.update(sync.stats)
            stats['state_final']=sync.state
            stats['last_frame']=sync.lastFrame
            stats['cpu_sync_only_ratio']=max(0.0,stats['cpu_total_sec']-stats['cpu_correct_wrong_sec'])/max(1e-9,stats['audio_in_sec'])
            stats['cpu_total_ratio']=stats['cpu_total_sec']/max(1e-9,stats['audio_in_sec'])
    finally:
        if player:
            if player.stdin:
                try:player.stdin.close()
                except BrokenPipeError:pass
            try:player.wait(timeout=3)
            except subprocess.TimeoutExpired:player.terminate();player.wait()
        correct.close();wrong.close()
        csvf.close();evf.close()
        (out/'summary.json').write_text(json.dumps(stats,indent=2)+'\n')
        print('RX SUMMARY:',json.dumps(stats,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description='LIVE speech A/B/C proof: cipher, correct index, wrong +64 index; NOT secure encryption')
    p.add_argument('--role',choices=['tx','rx'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--key-file',required=True,help='SAME binary key on TX and RX; >=16 bytes')
    p.add_argument('--session',default='voiceproof-1',help='Same TX/RX session')
    p.add_argument('--grain',type=int,default=80,help='PCM chunk samples: 80 (Opus-friendly) or 40 (strong scrambling)')
    p.add_argument('--wrong-offset',type=int,default=64,help='Only wrong RX branch uses this added absolute frame index')
    p.add_argument('--input',help='TX offline mono WAV, optional')
    p.add_argument('--generate-only',action='store_true',help='TX only: output WAV instead of starting Telemost')
    p.add_argument('--wav',help='RX offline: TimePack 20k mono WAV')
    p.add_argument('--duration',type=float,default=0.0,help='Live max seconds; 0 means Ctrl+C')
    p.add_argument('--mic-source',help='Physical mic PulseAudio source name, optional')
    p.add_argument('--speaker-sink',help='Physical headphones PulseAudio sink name, optional')
    p.add_argument('--listen',action='store_true',help='RX playback to real headphones')
    p.add_argument('--monitor',choices=['correct','cipher','wrong','rotate'],default='correct',help='Only live monitoring. All 3 WAV always saved')
    p.add_argument('--record-wire',action='store_true',help='TX: also write encrypted 20k TimePack WAV')
    p.add_argument('--tx-gain',type=float,default=0.75)
    p.add_argument('--guard-amp',type=float,default=0.12)
    p.add_argument('--io-samples',type=int,default=200)
    args=p.parse_args()
    if args.duration<0 or args.tx_gain<=0 or args.tx_gain>1 or args.guard_amp<=0 or args.guard_amp>=1:p.error('bad duration/gains')
    if args.role=='rx' and args.generate_only:p.error('--generate-only is TX only')
    if args.role=='tx' and args.wav:p.error('--wav is RX only')
    out=Path(args.out)
    if args.role=='tx':tx(args,out)
    else:rx(args,out)

if __name__=='__main__':main()