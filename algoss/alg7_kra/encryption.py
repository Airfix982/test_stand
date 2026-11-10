import numpy as np
import matplotlib.pyplot as plt
import math

class parameters_encryption:
	def __init__(self):
		self.pitch_ind_enc = 0
		self.energy_ind_enc = 0
		self.timbre_ind_enc = np.zeros(8, dtype = np.uint16)
		self.pitch_ind_dec = 0
		self.energy_ind_dec = 0
		self.timbre_ind_dec = np.zeros(8, dtype = np.uint16)	

class Encryption_class:
	def __init__(self, code_file):
		reloaded = np.load(code_file)
		self.generator_matrix = reloaded['generator_matrix']
		self.order_bits = reloaded['order_bits']
		self.G_dim_half = len(self.order_bits)
		self.G_dim = self.G_dim_half*2
		self.u_0 = reloaded['u_0']
		self.c_0 = reloaded['c_0']
		self.lattice_scale = reloaded['lattice_scale']
		self.boundary_box = reloaded['boundary_box']

		self.encryption_bits = reloaded['encryption_bits']
		self.quantization_levels = reloaded['quantization_levels']

	def select_vector(self, secret_key):
		v = np.zeros(8)
		k = 0
		for i in range(self.G_dim_half):
			bits = secret_key[k:k+self.order_bits[i]]
			k += self.order_bits[i]
			coeff = 0
			for bit in bits:
				coeff = (coeff << 1) | bit
			v += coeff*self.generator_matrix[:,i]
		return v

	def encipher_timbre(self, plaintext, secret_key):
		v = self.select_vector(secret_key)
		return np.mod(plaintext + v, self.boundary_box)

	def decipher_timbre(self, ciphertext, secret_key):
		v = self.select_vector(secret_key)
		return np.mod(ciphertext - v, self.boundary_box)

	def Encrypt_translation(self, param, keybits):
		key = 0;
		for bit in keybits:
			key = (key << 1) | bit
		return int((param + key) % self.quantization_levels)

	def Encrypt_rotation(self, param, keybits):
		param_enc = self.encipher_timbre(param, keybits)
		return param_enc

	def Decrypt_translation(self, param, keybits):
		key = 0;
		for bit in keybits:
			key = (key << 1) | bit
		return int((param - key) % self.quantization_levels)

	def Decrypt_rotation(self, param, keybits):	
		param_dec = self.decipher_timbre(param, keybits)
		return param_dec

	#######################

	def Speech_encryption(self, pitch_indices, energy_indices, timbre_indices, keybits):
		parameters = parameters_encryption()
		length = min(len(pitch_indices), len(energy_indices), len(timbre_indices[0,:]))
		pitch_indices_enc = np.zeros(length, dtype = "uint16")
		energy_indices_enc = np.zeros(length, dtype = "uint16")
		timbre_indices_enc = np.zeros([8, length], dtype = float)
		for i in range(length):
			self.Encrypt_frame(pitch_indices[i], energy_indices[i], timbre_indices[:,i], keybits[:,i], parameters)
			pitch_indices_enc[i]      = parameters.pitch_ind_enc
			energy_indices_enc[i]     = parameters.energy_ind_enc
			timbre_indices_enc[:,i]   = parameters.timbre_ind_enc
		return pitch_indices_enc, energy_indices_enc, timbre_indices_enc

	def Encrypt_frame(self, pitch_ind, energy_ind, timbre_ind, keybits, parameters):
		parameters.pitch_ind_enc  = self.Encrypt_translation(pitch_ind, keybits[0:self.encryption_bits[0]])
		parameters.energy_ind_enc = self.Encrypt_translation(energy_ind, keybits[self.encryption_bits[0]:self.encryption_bits[1]])
		parameters.timbre_ind_enc = self.Encrypt_rotation(timbre_ind, keybits[self.encryption_bits[1]:self.encryption_bits[2]])

	def encrypt_frame_rt(self, pitch_ind, energy_ind, timbre_ind, keybits):
		pitch_enc = self.Encrypt_translation(pitch_ind, keybits[0:self.encryption_bits[0]])
		energy_ind_enc = self.Encrypt_translation(energy_ind, keybits[self.encryption_bits[0]:self.encryption_bits[1]])
		timbre_ind_enc = self.Encrypt_rotation(timbre_ind, keybits[self.encryption_bits[1]:self.encryption_bits[2]])
		return pitch_enc, energy_ind_enc, timbre_ind_enc

	def speech_decryption(self, pitch_indices_rec, energy_indices_rec, timbre_indices_rec, keybits):
		parameters = parameters_encryption()
		length = min(len(pitch_indices_rec), len(energy_indices_rec), len(timbre_indices_rec[0,:]))
		pitch_indices_dec  = np.zeros(length)
		energy_indices_dec = np.zeros(length)
		timbre_indices_dec = np.zeros([8, length])
		for i in range(length):
			self.Decrypt_frame(pitch_indices_rec[i], energy_indices_rec[i], timbre_indices_rec[:,i], keybits[:,i], parameters)
			pitch_indices_dec[i]     = parameters.pitch_ind_dec
			energy_indices_dec[i]    = parameters.energy_ind_dec
			timbre_indices_dec[:,i]  = parameters.timbre_ind_dec
		return pitch_indices_dec, energy_indices_dec, timbre_indices_dec

	def Decrypt_frame(self, pitch_ind_rec, energy_ind_rec, timbre_ind_rec, keybits, parameters):
		parameters.pitch_ind_dec  = self.Decrypt_translation(pitch_ind_rec, keybits[0:self.encryption_bits[0]])
		parameters.energy_ind_dec = self.Decrypt_translation(energy_ind_rec, keybits[self.encryption_bits[0]:self.encryption_bits[1]])
		parameters.timbre_ind_dec = self.Decrypt_rotation(timbre_ind_rec, keybits[self.encryption_bits[1]:self.encryption_bits[2]])

	def decrypt_frame_rt(self, pitch_ind, energy_ind, timbre_ind, keybits):
		pitch_dec = self.Decrypt_translation(pitch_ind, keybits[0:self.encryption_bits[0]])
		energy_ind_dec = self.Decrypt_translation(energy_ind, keybits[self.encryption_bits[0]:self.encryption_bits[1]])
		timbre_ind_dec = self.Decrypt_rotation(timbre_ind, keybits[self.encryption_bits[1]:self.encryption_bits[2]])
		return pitch_dec, energy_ind_dec, timbre_ind_dec

