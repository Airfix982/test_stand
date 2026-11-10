import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as scp

class Speech_Synthesizer_class:
	def __init__(self, speech_model_file): # model): # PK LPCNet+boundaries
		reloaded = np.load(speech_model_file)
		# global parameters
		self.speech_frame_size = reloaded['speech_frame_size']
		self.quantization_levels = reloaded['quantization_levels']
		self.guard_ratio = reloaded['guard_ratio']
		self.nb_features = reloaded['nb_features']
		self.energy_min = reloaded['energy_min']
		self.energy_max = reloaded['energy_max']
		self.pitch_period_min = reloaded['pitch_period_min']
		self.pitch_period_max = reloaded['pitch_period_max']
		self.band_root_energy_min = reloaded['band_root_energy_min']
		self.band_root_energy_max = reloaded['band_root_energy_max']
		self.nb_features = reloaded['nb_features']

		self.guard_low  = self.quantization_levels*self.guard_ratio
		self.guard_high = self.quantization_levels - self.guard_low -1
		self.indices_range = self.quantization_levels - 2*self.guard_low

		# temprary parameters
		self.nb_frames = 0
		self.pitch_period_value = 0
		self.energy_value= 0
		self.band_energies = np.zeros(9)
		self.timbre_value = np.zeros(9)

	def spher2cart(self,point):
		x = np.ones([len(point)+1,1])
		for i in range(1,len(point)+1):
			x[i] = x[i-1]*np.sin(point[i-1])
		for i in range(len(point)):
			x[i] *= np.cos(point[i])
		return x

	def synthesize_speech(self, pitch_indices, energy_indices, timbre_angles):
		
		# compute speech features
		length = min(len(pitch_indices), len(energy_indices), len(timbre_angles[0,:]))
		pitch_period_values  = np.zeros(length)
		energy_values = np.zeros(length)
		timbre_values = np.zeros([9,length])
		band_energies_all = np.zeros([9,length])
		for i in range(length):
			self.index_to_value(pitch_indices[i], energy_indices[i], timbre_angles[:,i])
			pitch_period_values[i]    = self.pitch_period_value
			energy_values[i]          = self.energy_value
			timbre_values[:,i]        = self.timbre_value[:,0]
			band_energies_all[:,i]    = self.band_energies[:,0]

		# compute LPC coefficients from bands
		features = np.zeros((length,self.nb_features))
		features[:,9] = (pitch_period_values-100)/50
		for i in range(length-1):
			features[i,:9] = scp.dct(np.log10(band_energies_all[:,i]),type = 2,norm='ortho')

		features[:,0] -= 4
		features = np.reshape(features, (1, self.nb_features*length))

		features.astype('float32').tofile("temp/speech_features_dec_old.f32")
		os.system("./dump_data -decode temp/speech_features_dec_old.f32 temp/speech_features_dec_new.f32")
		features_new = np.fromfile("temp/speech_features_dec_new.f32", dtype='float32')

		features = np.reshape(features, (length, self.nb_features ))
		features_new = np.reshape(features_new, (length, self.nb_features ))

	def index_to_value(self, pitch_index, energy_index, timbre_angles):
		self.pitch_period_value = self.pitch_period_min + (self.pitch_period_max-self.pitch_period_min)*(min(max(pitch_index,self.guard_low ),self.guard_high)-self.guard_low )/self.indices_range
		self.energy_value = self.energy_min + (self.energy_max-self.energy_min)*(min(max(energy_index,self.guard_low ),self.guard_high)-self.guard_low )/self.indices_range
		for i in range(8):
			if timbre_angles[i] > np.pi/np.sqrt(8) :
				timbre_angles[i] = 2*np.pi/np.sqrt(8) - timbre_angles[i]
		self.timbre_value = self.spher2cart(timbre_angles*np.sqrt(8)/2)
		self.band_energies = np.power(10,self.energy_value)*np.square(self.timbre_value)