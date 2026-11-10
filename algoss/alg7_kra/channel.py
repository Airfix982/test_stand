import os
import numpy as np

def channel(signal, codec, Fs, rate = 0):

	signal = np.append(signal, np.zeros(160))
	signal.astype('int16').tofile("temp/signal_sent.raw")

	if codec == "AMR" and Fs == 8000:
		os.system("codecs/encoder_2018 MR122 temp/signal_sent.raw temp/AMR-enc.out")
		os.system("codecs/decoder_2018 temp/AMR-enc.out temp/signal_received.raw")
		signal_rec = np.fromfile("temp/signal_received.raw", dtype = 'int16')
		signal_rec = signal_rec[40:]
	elif codec == "SPEEX" and Fs == 8000:
		os.system("speexenc --bitrate " + str(rate) + " temp/signal_sent.raw temp/SPEEX-enc.out")
		os.system("speexdec temp/SPEEX-enc.out temp/signal_received.raw")
		signal_rec = np.fromfile("temp/signal_received.raw", dtype = 'int16')
	elif codec == "SILK" and (Fs == 8000 or Fs == 16000):
		os.system("codecs/opus_demo -e voip " + str(Fs) + " 1 " + str(rate) + " -cbr temp/signal_sent.raw temp/OPUS_SILK-enc.out")
		os.system("codecs/opus_demo -d " + str(Fs) + " 1 temp/OPUS_SILK-enc.out temp/signal_received.raw")
		signal_rec = np.fromfile("temp/signal_received.raw", dtype = 'int16')
		if Fs == 8000:
			signal_rec = signal_rec[53:]
		else:
			signal_rec = signal_rec[104:]
	elif codec == "CELT" and (Fs == 8000 or Fs == 16000):
		os.system("codecs/opus_demo -e audio " + str(Fs) + " 1 " + str(rate) + " -cbr temp/signal_sent.raw temp/OPUS_CELT-enc.out")
		os.system("codecs/opus_demo -d " + str(Fs) + " 1 temp/OPUS_CELT-enc.out temp/signal_received.raw")
		signal_rec = np.fromfile("temp/signal_received.raw", dtype = 'int16')
		if Fs == 8000:
			signal_rec = signal_rec[53:]
		else:
			signal_rec = signal_rec[104:]
	else:
		print("Bad channel parameters")
		signal_rec = [1]
	
	return signal_rec.astype('float')
