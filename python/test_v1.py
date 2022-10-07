import sys
from time import time

import numpy as np

import tensorflow as tf
tf.random.set_seed(8735)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from matplotlib import pyplot as plt

from my_monte_carlo import MonteCarlo
from my_quantum_system import *

def data_embedding(sample_init, x_train, npart, ndim=3):
    shifted_sample = []
    for i in range(len(x_train)):
        #gates.shift_gate(tf.ones(2*npart*ndim, dtype='double')*tf.constant(x_train)[i], npart, ndim)
        shifted_sample += [sample_init + tf.constant(x_train[i])]
    return shifted_sample



#
# Testing
#

# Superposisi ground state + 1st excited state

# Fungsi tebakan: dari solusi osilator harmonik kuantum (non-interacting)
def psi(rs, oms, deltaq, aa):
    al = 1
    bb = np.sqrt(1-(np.abs(aa))**2) # parameter excited state
    rt = rs - deltaq
    rexp = np.sum(oms*(rt)**2)
    return (aa*(1/(np.pi)**0.25)*np.exp(-0.5*al*rexp) + \
            bb*(np.prod(np.sqrt(2)*rt/(np.pi)**0.25)*np.exp(-0.5*al*rexp)))

psi_args = dict(
    oms=np.array([1.0, 1.0, 1.0]),
    deltaq=0.5,
    aa=1.0
)
# deltaq: terkait shifting gate (Killoran)

sampler = MonteCarlo(inseed=8735)

npart, ndim = 1, 3
print("number of particle:", npart, "\nnumber of dimension:", ndim)
print("omegas:", psi_args['oms'])

# Constants
N = 10000
strength = 0.0
m = 1.0

# Gezerlis

# Fermion (belum dibuktikan apakah akan bekerja atau tidak)
# Beberapa operasi gate tidak preserve fermionic symmetry

# Masalah di boson: tambah nonlin activation, energi lebih rendah dari true ground state

def bosonic_hamiltonian(q, p, npart, m, oms, strength):    
    # kinetic energy + one-body potential energy
    energy = 0.5*(p**2)/m + 0.5*m*((oms.reshape(1,1,-1)**2)*(q**2)) 
    energy = tf.math.reduce_sum(energy, axis=[1,2])
    energy = tf.math.reduce_mean(energy)
    # two-body interaction
    two_body_interaction = 0.0
    for k in range(npart):
        for j in range(k):
            two_body_interaction += tf.math.exp(-tf.math.reduce_sum((q[:, j, :] - q[:, k, :])**2, axis=1))
    two_body_interaction = strength*two_body_interaction
    
    # kinetic energy + one-body potential energy + two-body interaction
    return tf.math.reduce_mean(energy + two_body_interaction)


hamiltonian_args = dict(
    npart=npart,
    m=m,
    oms=psi_args['oms'],
    strength=strength
)


list_of_operations = [
    "beam_splitter", "rotate", "squeeze", "rotate", "beam_splitter", "shift", "fixed_single_cubic-0.01",
    "beam_splitter", "rotate", "squeeze", "rotate", "beam_splitter", "shift", "fixed_single_cubic-0.01",
    "beam_splitter", "rotate", "squeeze", "rotate", "beam_splitter", "shift", "fixed_single_cubic-0.01"]

list_of_bs_pair = [
    [ ['1x','1y'], ['1x','1z'], ['1y','1z'] ],
    [ ['1x','1y'], ['1x','1z'], ['1y','1z'] ],
    [ ['1x','1y'], ['1x','1z'], ['1y','1z'] ],
    [ ['1x','1y'], ['1x','1z'], ['1y','1z'] ],
    [ ['1x','1y'], ['1x','1z'], ['1y','1z'] ],
    [ ['1x','1y'], ['1x','1z'], ['1y','1z'] ]
]

# Object declaration
bosonic_system = quantum_system(
    N, npart, ndim, list_of_operations, list_of_bs_pair,
    psi, psi_args, bosonic_hamiltonian, hamiltonian_args,
    sampler,
    load_sample=True,
    q_dir='../ORIG_sandbox/saved_data/q_init_f1.txt',
    p_dir='../ORIG_sandbox/saved_data/p_init_f1.txt',
    inseed=8735
)
# load_sample=False: generate ulang distribusi q dan p


var_hist = []
H_hist = []

var_hist += [bosonic_system.variables.numpy()]
H_hist += [bosonic_system.hamiltonian_cost_function().numpy()]
H_min = np.min(H_hist)

lr = 0.001
delta_energy_min = 1e-6

opt = tf.keras.optimizers.Adam(learning_rate=lr)

delta_energy = 10
while delta_energy > delta_energy_min:
    step_count = opt.minimize(bosonic_system.hamiltonian_cost_function, [bosonic_system.variables]).numpy()
    var_hist += [bosonic_system.variables.numpy()]
    H_hist += [bosonic_system.hamiltonian_cost_function().numpy()]
    delta_energy = abs(H_hist[-1] - H_hist[-2])
    if step_count%200 == 0:
        print("Iteration:" + str(step_count) + \
            "\nEnergy:" + str(H_hist[-1]) + \
            ", Var:" + str(var_hist[-1]) + \
            ", Delta E:" + str(delta_energy))

print("\n")
print("Optimized variables:", var_hist[np.argmin(H_hist)])
print("Energy min:", np.min(H_hist))
print("Found in " + str(step_count) + " optimization steps.")

