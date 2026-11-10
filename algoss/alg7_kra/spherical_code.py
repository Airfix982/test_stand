import numpy as np
import matplotlib.pyplot as plt
import math

# here I define the code
k = 15    # bits
d = 2**k  # order

lattice_scale = 2*math.pi/np.sqrt(8*4*d**2)

boundary_box = 2*math.pi/np.sqrt(8)*np.ones(8)

generator_matrix = np.array([[2,-1, 0, 0, 0, 0, 0,0.5],\
                             [0, 1,-1, 0, 0, 0, 0,0.5],\
                             [0, 0, 1,-1, 0, 0, 0,0.5],\
                             [0, 0, 0, 1,-1, 0, 0,0.5],\
                             [0, 0, 0, 0, 1,-1, 0,0.5],\
                             [0, 0, 0, 0, 0, 1,-1,0.5],\
                             [0, 0, 0, 0, 0, 0, 1,0.5],\
                             [0, 0, 0, 0, 0, 0, 0,0.5]])
generator_matrix *= lattice_scale

angles = np.zeros([8,8])
angles[0,0] = 1
angles[0:2,1] = [-0.5, 0.5] 
angles[1:3,2] = [-0.5, 0.5] 
angles[2:4,3] = [-0.5, 0.5] 
angles[3:5,4] = [-0.5, 0.5] 
angles[4:6,5] = [-0.5, 0.5] 
angles[5:7,6] = [-0.5, 0.5] 
angles[:,7] = [1/4]
angles = angles*2*math.pi/d

G_gen = np.zeros([8,16,16])
for i in range(8):
	for j in range(8):
		G_gen[i,2*j:2*(j+1),2*j:2*(j+1)] = [[np.cos(angles[j,i]), -1*np.sin(angles[j,i])], \
		                                   [np.sin(angles[j,i]), np.cos(angles[j,i])]]

m = k+6*(k+1)+k+2;
G_all = np.zeros([m,16,16])

for i in range(k):
	G_all[i,:,:] = np.linalg.matrix_power(np.squeeze(G_gen[0,:,:]),2**i)
for i in range(6):
	for j in range(k+1):
		G_all[k+i*(k+1)+j,:,:] = np.linalg.matrix_power(np.squeeze(G_gen[i+1,:,:]),2**j)
for i in range(k+2):
	G_all[k+6*(k+1)+i,:,:] = np.linalg.matrix_power(np.squeeze(G_gen[7,:,:]),2**i)

V_all = np.zeros([8,m])
for i in range(k):
	V_all[:,i] = angles[:,0]*2**i
for i in range(6):
	for j in range(k+1):
		V_all[:,k+i*(k+1)+j] = angles[:,i+1]*2**j
for i in range(k+2):
	V_all[:,k+6*(k+1)+i] = angles[:,7]*2**i

c_0 = np.ones([8,1])/np.sqrt(8)
u_0 = np.zeros([16,1]);
for i in range(8):
	u_0[2*i,0] = c_0[i,0]

order_groups = np.array([d,2*d,2*d,2*d,2*d,2*d,2*d,4*d])
order_bits = np.array([k,k+1,k+1,k+1,k+1,k+1,k+1,k+2])

np.savez('spherical_code',  G_gen = G_gen, u_0 = u_0, c_0 = c_0, lattice_scale = lattice_scale, G_all = G_all, \
	V_all = V_all, order_groups = order_groups, order_bits = order_bits, boundary_box = boundary_box,  \
	generator_matrix = generator_matrix, encryption_bits = np.array([16,32,160]), quantization_levels = 2**16)