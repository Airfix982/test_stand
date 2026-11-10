# speech model parameters

import numpy as np

speech_frame_size = 160
nb_features = 23
feature_chunk_size = 15

energy_min = 1 # log10 scale
energy_max = 8

pitch_period_min = 16
pitch_period_max = 128 # pitch_period

band_root_energy_min = 1 # linear scale, corresponds to 0 = log10(1**2)
band_root_energy_max = 10e4 # corresponds to 8 = log10(10**4**2)

np.savez(
    'speech_model',
    speech_frame_size=speech_frame_size,
    nb_features=nb_features,
    energy_min=energy_min,
    energy_max=energy_max,
    pitch_period_min=pitch_period_min,
    pitch_period_max=pitch_period_max,
    band_root_energy_min=band_root_energy_min,
    band_root_energy_max=band_root_energy_max,
    quantization_levels=2**16,
    guard_ratio=0.125
)