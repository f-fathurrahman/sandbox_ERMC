import numpy as np

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from my_utils import utils

class QuantumSystem():

    def __init__(self, N, npart, ndim,
               list_of_operations, list_of_bs_pair=[],
               psi=None, psi_args=None,
               hamiltonian=None, hamiltonian_args=None,
               sampler=None, load_sample=False, q_dir=None, p_dir=None,
               var_init=None, inseed=None):
        #
        self.N = N
        self.npart = npart
        self.ndim = ndim
        self.inseed = inseed
        self.list_of_operations = list_of_operations
        if list_of_operations.count("beam_splitter") != len(list_of_bs_pair):
            sys.exit("There are " + str(list_of_operations.count("beam_splitter")) + 
                     " 'beam_splitter' operation(s) but only " + str(len(list_of_bs_pair)) + " list of pair given.")
        else:
            self.list_of_bs_pair = utils.xyz_to_numbers(list_of_bs_pair, ndim)
        #
        self.psi = psi
        self.psi_args = psi_args
        #
        self.hamiltonian = hamiltonian
        self.hamiltonian_args = hamiltonian_args
        #
        # Initialize variables
        self.nvar = utils.calculate_variables(self.list_of_operations, self.list_of_bs_pair, npart, ndim)
        try:
            if var_init == None:
                np.random.seed(self.inseed)
                random_initial_variables = 1.0 - 0.1*np.random.rand(self.nvar)
                self.variables = tf.Variable(random_initial_variables, trainable=True, dtype='double')
            else:
                self.variables = tf.Variable(var_init, trainable=True, dtype='double')
        except:
            if len(list(var_init)) != self.nvar:
                 sys.exit("The length of var_init must equal to " + str(self.nvar) + ".")
            else:
                self.variables = tf.Variable(var_init, trainable=True, dtype='double')
        #
        # Initialize samples
        if load_sample == False:
            _, _, self.p_init, self.q_init = sampler.sampling(psi, psi_args, N, npart, ndim)
        else:
            self.q_init = np.loadtxt(q_dir).reshape(N, npart, ndim)
            self.p_init = np.loadtxt(p_dir).reshape(N, npart, ndim)
        #
        self.q_init = tf.convert_to_tensor(self.q_init, dtype='double')
        self.p_init = tf.convert_to_tensor(self.p_init, dtype='double')


    @tf.function
    def operators_init(self):
        return utils.initialize_gates(self.list_of_operations, self.list_of_bs_pair, self.variables, self.npart, self.ndim)
  

    @tf.function
    def hamiltonian_cost_function(self):
        # Initialize operators
        self.operators = self.operators_init()
        # Apply transformations
        self.q_new, self.p_new = utils.qp_transformation(
            self.list_of_operations, self.list_of_bs_pair,
            self.operators,
            self.q_init, self.p_init,
            self.N, self.npart, self.ndim)
        # Will return a hamiltonian
        return self.hamiltonian(self.q_new, self.p_new, **self.hamiltonian_args)
