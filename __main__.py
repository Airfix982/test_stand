import argparse
from config import AppConfig, CheckerConfig, DeviceConfig, AudioSpec
from runner import Runner
from analysis.alignment import analyze
from experiment.channelMesurment import ChannelMeasurementRunner

def addCommon(parser):
    defaults = DeviceConfig()
    parser.add_argument("--channel", default="telemost", choices=["telemost", "ds", "tg"])
    parser.add_argument("--micro", default=defaults.micro)
    parser.add_argument("--headph", default=defaults.headphones)
    parser.add_argument("--rate", type=int, default=44100)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--block", type=int, default=1024)
    parser.add_argument("--recdir", help="dir where records are stored", default="records")
    parser.add_argument("--iface", default="", help="network interface; empty = detect default")


def main():
    defaults = DeviceConfig()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    addCommon(run)
    run.add_argument("--role", choices=["tx", "rx", "full"], default="full")
    run.add_argument("--mode", choices=["live", "probe"], default="live")
    run.add_argument("--duration", type=float, default=60.0)
    run.add_argument("--seed", type=int, default=4444)
    run.add_argument("--probe-block", type=float, default=1.0)
    run.add_argument("--probe-level", type=float, default=-15.0)
    measure = sub.add_parser("measure")
    addCommon(measure)
    measure.add_argument("--role", choices=["tx", "rx"], required=True)
    measure.add_argument("--duration", type=float, required=True)
    measure.add_argument("--seed", type=int, default=4444)
    measure.add_argument("--probe-block", type=float, default=1.0)
    measure.add_argument("--probe-level", type=float, default=-15.0)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--rx", required=True)
    analysis.add_argument("--out", required=True)
    analysis.add_argument("--seed", type=int, default=4444)
    analysis.add_argument("--probe-block", type=float, default=1.0)
    analysis.add_argument("--probe-level", type=float, default=-15.0)
    analysis.add_argument("--initial-search", type=float, default=10.0)
    analysis.add_argument("--search-ms", type=float, default=150.0)
    analysis.add_argument("--tx", default=None)
    analysis.add_argument("--tx-duration", type=float, default=None)
    analysis.add_argument("--event-ms", type=float, default=2.0)
    analysis.add_argument("--event-corr-min", type=float, default=0.20)
    analysis.add_argument("--reference-blocks", type=int, default=5)
    
    args = parser.parse_args()
    if args.cmd == "analyze":
        summary, csv_path = analyze(
            rxPath=args.rx, outDir=args.out,
            seed=args.seed, blockSec=args.probe_block, dbfsLvl=args.probe_level,
            initialSearch=args.initial_search, searchMs=args.search_ms,
            txPath=args.tx, txDuration=args.tx_duration,
            eventMs=args.event_ms, eventCorrMin=args.event_corr_min,
            referenceBlocks=args.reference_blocks
        )
        print(summary)
        print(csv_path)
        return
    audio = AudioSpec(rate=args.rate, channels=args.channels, ioBlockSamples=args.block)
    devices = DeviceConfig(micro=args.micro, headphones=args.headph)
    checker = CheckerConfig(role=args.role, mode=args.mode, durationSec=args.duration, seed=args.seed, probeBlockSec=args.probe_block, probeLvlDbfs=args.probe_level, interface=args.iface)


    audio = AudioSpec(rate=args.rate, channels=args.channels, ioBlockSamples=args.block)
    devices = DeviceConfig(micro=args.mic, headphones=args.headph)

    if args.cmd == "measure":
        checker = CheckerConfig(
            role=args.role, mode="probe", durationSec=args.duration,
            seed=args.seed, probeBlockSec=args.probe_block,
            probeLvlDbfs=args.probe_level, interface=args.iface
        )

        cfg = AppConfig(
            channel=args.channel, recordsDir=args.recdir,
            audio=audio, devices=devices, checker=checker
        )

        ChannelMeasurementRunner(cfg).run()
        return
    
    
    Runner(AppConfig(channel = args.channel, recordsDir=args.recdir, audio=audio, devices=devices, checker=checker)).run()

if __name__ == "__main__":
    main()
