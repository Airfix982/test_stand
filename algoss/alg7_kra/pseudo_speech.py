import numpy as np
import math
import matplotlib.pyplot as plt
from .single_pitch import single_pitch

class parameters_pseudospeech:
	def __init__(self, band_num, harm_num_max):
		self.pitch_index = 0
		self.energy_index = 0
		self.timbre_index = np.zeros(int(band_num/2), dtype = float)
		self.timbre_value = np.zeros(band_num, dtype = float)
		self.pitch_value = 0
		self.pitch_value_prev = 0
		self.energy_value = 0
		self.envelope_vector = np.zeros(band_num, dtype = float)		
		self.harm_num = 0
		self.harm_num_prev = harm_num_max	
		self.phases = np.zeros(harm_num_max, dtype = float) 	
		self.phases_prev = np.zeros(harm_num_max, dtype = float)
		self.amplitudes = np.zeros(harm_num_max, dtype = float)
		self.amplitudes_prev = np.zeros(harm_num_max, dtype = float)

class Pseudospeech_Synthetizer_class:
	def __init__(self, pseudospeech_model_file, spherical_code_file):
		reloaded = np.load(pseudospeech_model_file)
		self.Fs = int(reloaded['Fs'])
		self.flng = int(reloaded['flng'])
		self.glng = int(reloaded['glng'])
		self.quantization_levels = reloaded['quantization_levels']
		self.band_num = int(reloaded['band_num'])	
		self.pitch_period_min = reloaded['pitch_period_min']
		self.pitch_period_max = reloaded['pitch_period_max']
		self.energy_min = reloaded['energy_min']
		self.energy_max = reloaded['energy_max']
		self.FB_pinv_all = reloaded['FB_pinv_all']
		self.filter_bank = reloaded['filter_bank']
		self.cutoff_freq_min = reloaded['cutoff_frequency_min']
		self.cutoff_freq_max = reloaded['cutoff_frequency_max']


		reloaded_code = np.load(spherical_code_file)
		self.c_0 = np.asarray(reloaded_code["c_0"], dtype=np.float64).reshape(-1)
		self.u_0 = np.asarray(reloaded_code["u_0"], dtype=np.float64).reshape(-1)
		self.order_bits = reloaded_code['order_bits']
		self.G_dim_half = len(self.order_bits)
		self.G_dim = self.G_dim_half*2

		self.pitch_min = 1/self.pitch_period_max*self.Fs
		self.pitch_max = 1/self.pitch_period_min*self.Fs
		self.harm_num_max = int(self.cutoff_freq_max/self.pitch_min)
		self.cutoff_max = int(self.cutoff_freq_max/self.Fs*self.flng)

	def compute_complex_amplitudes(self, FB_pinv, envelope_vector):
		envelope_vector = envelope_vector*np.exp(1j*2*math.pi*np.array(range(self.band_num))/self.band_num)
		A = np.matmul(FB_pinv,envelope_vector)
		return np.abs(A), np.angle(A)

	def torus_map(self,point):
		u = np.zeros([self.G_dim])
		for i in range(self.G_dim_half):
			u[2*i:2*i+2] = [self.c_0[i]*np.cos(point[i]/self.c_0[i]), \
			                  self.c_0[i]*np.sin(point[i]/self.c_0[i])]
		return u

	def torus_map_rev(self,point):
		u = np.zeros([self.G_dim_half])
		for i in range(self.G_dim_half):
			u[i] = np.arccos(point[2*i]/self.c_0[i])*self.c_0[i]
			if point[2*i+1] < 0: u[i] = 2*math.pi*self.c_0[i]-u[i]
		return u

	def projection(self, point):
		y = np.zeros(self.G_dim)
		for i in range(self.G_dim_half):
			point_norm = max(np.linalg.norm(point[2*i:2*i+2]),1e-6)
			y[2*i:2*i+2] = point[2*i:2*i+2]*self.u_0[2*i]/point_norm
		return y	

	def index_to_value(self, pitch_index, energy_index, timbre_index, parameters):
		parameters.pitch_index  = pitch_index
		parameters.energy_index = energy_index
		parameters.timbre_index = timbre_index
		parameters.pitch_value  = self.Fs/(self.pitch_period_min + (self.pitch_period_max-self.pitch_period_min)*pitch_index/self.quantization_levels)
		parameters.energy_value = np.power(10,self.energy_min + (self.energy_max-self.energy_min)/self.quantization_levels*energy_index)
		parameters.timbre_value = self.torus_map(timbre_index)
		parameters.harm_num = int(self.cutoff_max*self.Fs/self.flng/parameters.pitch_value)
		parameters.envelope_vector = parameters.timbre_value

	def value_to_index(self, parameters):
		parameters.pitch_index = max(min(np.round((self.Fs/parameters.pitch_value-self.pitch_period_min)/(self.pitch_period_max-self.pitch_period_min)*self.quantization_levels),self.quantization_levels-1),0)
		# parameters.energy_index = max(min(np.round((np.log10(parameters.energy_value)-self.energy_min)/(self.energy_max-self.energy_min)*self.quantization_levels),self.quantization_levels-1),0)
		energy_value = max(float(parameters.energy_value), 1e-12)
		parameters.energy_index = max(
			min(
				np.round(
					(np.log10(energy_value) - self.energy_min)
					/ (self.energy_max - self.energy_min)
					* self.quantization_levels
				),
				self.quantization_levels - 1
			),
			0
		)
		parameters.timbre_index = self.torus_map_rev(self.projection(parameters.timbre_value))

	###########################				

	def analyze_pseudospeech(self, input_signal):
		parameters = parameters_pseudospeech(self.band_num, self.harm_num_max)
		f0Estimator = single_pitch(self.flng, self.harm_num_max, np.array([self.pitch_min, self.pitch_max])/self.Fs)
		length = int(len(input_signal)//(self.flng+self.glng))
		energy_indices = np.zeros(length)
		pitch_indices  = np.zeros(length)
		timbre_indices = np.zeros([self.G_dim_half,length])
		#######
		energy_values = np.zeros(length)
		pitch_values  = np.zeros(length)
		timbre_values = np.zeros([self.G_dim,length])
		#######
		for i in range(length):
			self.analyze_frame(input_signal[i*(self.flng+self.glng):(i+1)*(self.flng+self.glng)], f0Estimator, parameters)
			pitch_indices[i]    = parameters.pitch_index
			energy_indices[i]   = parameters.energy_index
			timbre_indices[:,i] = parameters.timbre_index
			#######
			pitch_values[i]    = self.Fs/parameters.pitch_value
			energy_values[i]   = parameters.energy_value
			timbre_values[:,i] = parameters.timbre_value
			#######

		return pitch_indices, energy_indices, timbre_indices

	def analyze_frame(self, input_frame, f0Estimator, parameters):
		x = np.asarray(input_frame[self.glng:], dtype=np.float64)
		X = np.fft.fft(x)

		phase = np.exp(-1j * 2 * np.pi * np.arange(self.band_num) / self.band_num)
		parameters.envelope_vector = 2 * np.real((X[:len(X) // 2] @ self.filter_bank.T) * phase)
		parameters.energy_value = max(float(np.dot(x, x)), 1e-12)

		norm = max(np.linalg.norm(parameters.envelope_vector), 0.001)
		parameters.timbre_value = parameters.envelope_vector / norm
		parameters.pitch_value = (self.Fs / (2 * np.pi)) * f0Estimator.est(x, eps=2e-2)

		self.value_to_index(parameters)
		return parameters.pitch_index, parameters.energy_index, parameters.timbre_index

	# def analyze_frame(self, input_frame, f0Estimator, parameters):
	# 	X = np.fft.fft(input_frame[self.glng:])
	# 	parameters.envelope_vector = 2*np.real(np.matmul(X[:len(X)//2],np.transpose(self.filter_bank))*np.exp(-1j*2*math.pi*np.array(range(self.band_num))/self.band_num))
	# 	parameters.energy_value = np.sum(np.square(input_frame[self.glng:]))
	# 	parameters.timbre_value = parameters.envelope_vector/max(np.linalg.norm(parameters.envelope_vector),0.001)
	# 	parameters.pitch_value = (self.Fs/(2*np.pi))*f0Estimator.est(np.array(input_frame[self.glng:], dtype=np.float64), eps=2*1e-2)
	# 	self.value_to_index(parameters)

	# 	return parameters.pitch_index, parameters.energy_index, parameters.timbre_index, 

	def synthesize_pseudospeech(self, pitch_indices, energy_indices, timbre_indices):
		parameters = parameters_pseudospeech(self.band_num, self.harm_num_max)
		length = min(len(pitch_indices), len(energy_indices), len(timbre_indices[0,:]))
		signal = np.zeros(length*(self.flng+self.glng))
		#######
		energy_values = np.zeros(length)
		pitch_values  = np.zeros(length)
		timbre_values = np.zeros([self.G_dim,length])
		#######
		for i in range(length):
			signal[i*(self.flng+self.glng):(i+1)*(self.flng+self.glng)] = self.synthetize_frame(pitch_indices[i], energy_indices[i], timbre_indices[:,i], parameters)

		return signal

	def synthetize_frame(self, pitch_index, energy_index, timbre_index, parameters):

		# parameters computation
		self.index_to_value(pitch_index, energy_index, timbre_index, parameters)
		FB_pinv = self.FB_pinv_all[pitch_index,0:parameters.harm_num,:]
		parameters.amplitudes, parameters.phases = self.compute_complex_amplitudes(FB_pinv, parameters.envelope_vector)

		#guard period
		amplitudes_guard_prev  = np.linspace(parameters.amplitudes_prev[0:parameters.harm_num_prev],np.zeros(parameters.harm_num_prev),self.glng)
		pulsation_prev = parameters.pitch_value_prev*np.linspace(1,parameters.harm_num_prev,parameters.harm_num_prev)*2*math.pi/self.flng
		phases_guard = np.multiply(np.linspace(np.ones(parameters.harm_num_prev),self.glng*np.ones(parameters.harm_num_prev),self.glng),pulsation_prev) + np.transpose(parameters.phases_prev[0:parameters.harm_num_prev])
		sines_guard  = np.multiply(np.cos(phases_guard),amplitudes_guard_prev)
		guard        = np.sum(sines_guard,1)

		#frame with guard
		amplitudes_guard = np.linspace(np.zeros(parameters.harm_num),parameters.amplitudes[0:parameters.harm_num],self.glng)
		amplitudes_frame = np.array([parameters.amplitudes[0:parameters.harm_num],]*self.flng) 
		amplitudes_g_frame = np.concatenate((amplitudes_guard, amplitudes_frame))
		pulsation = parameters.pitch_value*np.linspace(1,parameters.harm_num,parameters.harm_num)*2*math.pi/self.Fs
		phases_frame = np.multiply(np.linspace(-1*self.glng*np.ones(parameters.harm_num),(self.flng-1)*np.ones(parameters.harm_num),(self.glng+self.flng)),pulsation) + np.transpose(parameters.phases[0:parameters.harm_num])
		sines_frame = np.multiply(np.cos(phases_frame),amplitudes_g_frame)
		frame = np.sum(sines_frame,1)

		energy = np.sum(np.square(frame[self.glng:]))
		frame *= np.sqrt(parameters.energy_value/energy) 
		parameters.amplitudes *= np.sqrt(parameters.energy_value/energy)

		#post processing
		parameters.amplitudes_prev[:parameters.harm_num] = parameters.amplitudes[:parameters.harm_num]
		parameters.harm_num_prev = parameters.harm_num
		parameters.pitch_value_prev = parameters.pitch_value
		parameters.phases_prev[0:parameters.harm_num] = phases_frame[-1,:]
		frame[0:self.glng] = frame[0:self.glng] + guard
		
		return np.clip(frame,-32760,32760).astype('int16')	