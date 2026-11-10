import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as scp

from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
SPEECH_ENCODER = BASE_DIR / "speech_encoder"

class Speech_Analyzer_class:
	def __init__(self, speech_model_file, spherical_code_file): # model): # PK LPCNet+boundaries
		reloaded = np.load(speech_model_file)
		# global parameters
		self.speech_frame_size = reloaded['speech_frame_size']
		self.nb_features = reloaded['nb_features']
		self.quantization_levels = reloaded['quantization_levels']
		self.guard_ratio = reloaded['guard_ratio']
		self.energy_min = reloaded['energy_min']
		self.energy_max = reloaded['energy_max']
		self.pitch_period_min = reloaded['pitch_period_min']
		self.pitch_period_max = reloaded['pitch_period_max']
		self.band_root_energy_min = reloaded['band_root_energy_min']
		self.band_root_energy_max = reloaded['band_root_energy_max']

		self.guard_low  = self.quantization_levels*self.guard_ratio
		self.guard_high = self.quantization_levels - self.guard_low -1
		self.indices_range = self.quantization_levels - 2*self.guard_low

		reloaded = np.load(spherical_code_file)
		self.lattice_scale = reloaded['lattice_scale']

		# temprary parameters
		self.nb_frames = 0
		self.pitch_index = 0
		self.energy_index = 0
		self.timbre_index = np.zeros([8,1])

	def cart2spher(self, point):
		fi = np.zeros([len(point)-1])
		for i in range(len(point)-2):
			p_norm = np.linalg.norm(point[i:])
			if p_norm == 0:
				fi[i] = 0
			else:
				fi[i] = np.arccos(point[i]/p_norm)
		p_norm = np.linalg.norm(point[-2:])
		if p_norm != 0:
			if point[-1] > 0:
				fi[-1] = np.arccos(point[-2]/p_norm)
			else:
				fi[-1] = 2*np.pi - np.arccos(point[-2]/p_norm)
		return fi

	def closest_vector_Gamma8(self, point):
		x = point/self.lattice_scale
		a = np.round(x)
		if np.sum(a) % 2 == 1:
			id_a = np.argmax(np.abs(a-x))
			if a[id_a] - x[id_a] > 0:
				a[id_a] -= 1
			else:
				a[id_a] += 1
		b = np.round(x-0.5)
		if np.sum(b) % 2 == 1:
			id_b = np.argmax(np.abs(b-x+0.5))
			if b[id_b] - x[id_b] + 0.5 > 0:
				b[id_b] -= 1
			else:
				b[id_b] += 1
		b += 0.5
		if np.sum(np.square(a-x)) < np.sum(np.square(b-x)):
			c = a
		else: 
			c = b
		return c*self.lattice_scale

	def analyze_speech(self, speech_samples):
		TEMP_DIR.mkdir(parents=True, exist_ok=True)

		samples_file = TEMP_DIR / "speech_samples.i16"
		features_file = TEMP_DIR / "speech_features_temp.f32"

		speech_samples.tofile(samples_file)

		subprocess.run([str(SPEECH_ENCODER), "-test", str(samples_file), str(features_file), ], check=True)

		features = np.fromfile(features_file, dtype="float32")

		self.nb_frames = len(features) // self.nb_features
		features = features[:self.nb_frames * self.nb_features]
		features = np.reshape(features, [self.nb_frames, self.nb_features])
		self.last_features = features.copy()

		features = features[:self.nb_frames*self.nb_features]
		features = np.reshape(features,[self.nb_frames, self.nb_features])
		pitch_period_values = features[:,9]*50+100

		mfcc = features[:,:9]
		mfcc[:,0] += 4
		band_energies = np.zeros((self.nb_frames,9))
		for i in range(self.nb_frames):
			band_energies[i,:] = scp.idct(mfcc[i,:],type = 2,norm='ortho')
		band_energies = np.power(10,band_energies)
		energy_values = np.log10(np.sum(band_energies,axis = 1))
		band_root_energy_values = np.sqrt(band_energies)
		
		# encode parameters
		timbre_angles = np.zeros([8, len(energy_values)])
		timbre_values = np.zeros([9, len(energy_values)])
		for i in range(len(energy_values)):
			timbre_values[:,i] = band_root_energy_values[i,:]/np.linalg.norm(band_root_energy_values[i,:])
			timbre_angles[:,i] = self.cart2spher(band_root_energy_values[i,:]/np.linalg.norm(band_root_energy_values[i,:]))
				
		pitch_indices  = np.zeros(int(self.nb_frames))
		energy_indices = np.zeros(int(self.nb_frames))
		timbre_indices = np.zeros([8, (int(self.nb_frames))])
		for i in range((int(self.nb_frames))):
			self.value_to_index(pitch_period_values[i], energy_values[i], timbre_angles[:,i])
			pitch_indices[i]    = self.pitch_index
			energy_indices[i]   = self.energy_index
			timbre_indices[:,i] = self.timbre_index

		return pitch_indices.astype('uint16'), energy_indices.astype('uint16'), timbre_indices, np.clip(pitch_period_values,self.pitch_period_min,self.pitch_period_max), np.power(10,np.clip(energy_values,self.energy_min,self.energy_max)), timbre_values

	def value_to_index(self, pitch_period_value, energy_value, timbre_angles):
		self.pitch_index  = max(min(np.round((pitch_period_value-self.pitch_period_min)/(self.pitch_period_max-self.pitch_period_min)*self.indices_range)+self.guard_low,self.guard_high),self.guard_low)
		self.energy_index = max(min(np.round((energy_value-self.energy_min)/(self.energy_max-self.energy_min)*self.indices_range)+self.guard_low,self.guard_high),self.guard_low)
		self.timbre_index = self.closest_vector_Gamma8(2*timbre_angles/np.sqrt(8)) # scaling by two from [0,pi/2] to [0,pi]

	def indices_to_values(self, pitch_index, energy_index, timbre_index):
		pitch_period = self.pitch_period_min + ( (float(pitch_index) - self.guard_low) / self.indices_range ) * (self.pitch_period_max - self.pitch_period_min)

		energy_log = self.energy_min + ( (float(energy_index) - self.guard_low) / self.indices_range ) * (self.energy_max - self.energy_min)

		energy = 10.0 ** energy_log

		angles = np.asarray(timbre_index, dtype=np.float64) * np.sqrt(8.0) / 2.0

		timbre = np.zeros(9, dtype=np.float64)

		prod = 1.0

		for i in range(8):
			timbre[i] = prod * np.cos(angles[i])
			prod *= np.sin(angles[i])

		timbre[8] = prod

		n = np.linalg.norm(timbre)
		if n > 1e-12:
			timbre /= n

		return pitch_period, energy, timbre