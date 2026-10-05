import argparse
from config import AppConfig, DeviceConfig, AudioSpec
from runner import Runner

def main():
    defaults = DeviceConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="telemost")
    parser.add_argument("--micro", default=defaults.micro)
    parser.add_argument("--headph", default=defaults.headphones)
    parser.add_argument("--rate", type=int, default=44100)
    parser.add_argument("--channels", default=1, type=int)
    parser.add_argument("--block", default=1024, type=int)
    parser.add_argument("--recdir", help="dir where records are stored", default="records")
    args = parser.parse_args()
    audio = AudioSpec(rate=args.rate, channels=args.channels, ioBlockSamples=args.block)
    devices = DeviceConfig(micro=args.micro, headphones=args.headph)
    Runner(AppConfig(channel = args.channel, recordsDir=args.recdir, audio=audio, devices=devices)).run()

if __name__ == "__main__":
    main()
