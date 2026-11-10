import numpy as np
import matplotlib.pyplot as plt
import math

Fs = 16000 # Hz
frame_length = 240 # samples
guard_length = 80 # samples 
cutoff_frequency_min = 300 # Hz (the lowest frequency) 
band_num = 16

pitch_period_min = 80  # samples /200 Hz
pitch_period_max = 160 # samples /100  Hz
pitch_period_num = 2**16

energy_min = 9 
energy_max = 10

### different filter banks ###

## trapezoidal wide with freq_shift 300 Hz
# filter_bank = np.zeros([band_num, frame_length//2], dtype = float) # my filterbank
# for i in range(band_num):
#   filter_bank[i, range(i*6+4,(i+1)*6+4+1)] = np.transpose([0.5,1,1,1,1,1,0.5])
# cutoff_max = band_num*6+4

## square wide with freq_shift 300 Hz
filter_bank = np.zeros([band_num, frame_length//2], dtype = float) # my filterbank
for i in range(band_num):
  filter_bank[i, range(i*6+4,(i+1)*6+4)] = np.transpose([1,1,1,1,1,1])
cutoff_max = band_num*6+4
cutoff_frequency_max = cutoff_max*Fs/frame_length
cutoff_min = cutoff_frequency_min/Fs*frame_length

pitch_period_values = pitch_period_min+np.array(range(pitch_period_num))/pitch_period_num*(pitch_period_max-pitch_period_min)

pitch_values = 1/pitch_period_values*Fs

harm_num_max = int(cutoff_frequency_max/np.min(pitch_values))
FB_pinv_all = np.zeros([pitch_period_num, harm_num_max, band_num], dtype = complex)
for index in range(len(pitch_values)):
    if index % 256 == 0:
        print(f"{index}/{len(pitch_values)} ({100.0 * index / len(pitch_values):.1f}%)", flush=True)
    pitch = pitch_values[index]
    harm_num = int(cutoff_frequency_max / pitch)
    B = np.zeros([frame_length // 2, harm_num], dtype=complex)

    for i in range(harm_num):
        if pitch * (i + 1) > cutoff_min + 0.5:
            t = np.exp(1j * 2 * math.pi * np.array(range(frame_length)) * (i + 1) * pitch / Fs)
            T = np.fft.fft(t)
            B[:, i] = T[0:frame_length // 2]
    FB = np.matmul(filter_bank, B)
    FB_pinv_all[index, 0:harm_num, :] = np.linalg.pinv(FB)

np.savez('pseudospeech_model', Fs = Fs, flng = int(frame_length), glng = int(guard_length),  \
	band_num = int(band_num),  FB_pinv_all = FB_pinv_all, filter_bank = filter_bank, pitch_period_min = pitch_period_min,\
	pitch_period_max = pitch_period_max, energy_min = energy_min, energy_max = energy_max, \
	cutoff_frequency_min = cutoff_frequency_min, cutoff_frequency_max = cutoff_frequency_max, \
	quantization_levels = 2**16, guard_ratio = 0.125)
