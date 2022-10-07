import sys
from time import time

import numpy as np
import copy

import tensorflow as tf
tf.random.set_seed(8735)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from matplotlib import pyplot as plt


class monte_carlo():
  
    def __init__(self, h=0.01, inseed=None):
        self.h = h
        self.inseed = inseed

    def rho(self, psi, rs, psi_args):
        rhof = (np.conj(psi(rs, **psi_args)))*psi(rs, **psi_args)
        return rhof

    def ERmom(self, psi, rs, psi_args, s):
        npart, ndim = rs.shape
        rhoold2 = self.rho(psi, rs, psi_args)
        p1 = complex(0.,0.)
        p2 = 0.0
        p_arr = np.zeros((npart,ndim),dtype=complex)
        hbar = 1.0
        for ipart in range(npart):
            dif1 = 0.0
            dif_sqrho1 = 0.0
            psi0t = 0.0
            sqrho0t = 0.0
            dif = 0.0
            dif_conj = complex(0.,0.)
            dif_rho = 0.0
            for idim in range(ndim):
                r = rs[ipart,idim]
                rs[ipart,idim] = r + self.h
                psip = psi(rs, **psi_args)                  #p2
                psip_conj = np.conj(psi(rs, **psi_args))     #p2
                rhop = self.rho(psi, rs, psi_args)       #p1
                sqrhop = np.sqrt(rhop)
                rs[ipart,idim] = r
                psi0 = psi(rs, **psi_args)                 #p2
                psi0_conj = np.conj(psi(rs, **psi_args))     #p2
                rho0 = self.rho(psi, rs, psi_args)       #p1
                dif = (psip-psi0)/self.h                   #p2
                dif_conj = (psip_conj-psi0_conj)/self.h    #p2
                dif_rho = (rhop-rho0)/self.h   #p1          
                p1 = (0.5*hbar/complex(0,1))*((dif/psi0)-(dif_conj/psi0_conj))
                p2 = s*0.5*(dif_rho/rhoold2) #eq. 12, Budiyono 2020
                p_arr[ipart,idim] = (p1+p2)
        return p_arr                 #momentum (added for calculating Heisenberg UR)

    def sampling(self, psi, psi_args, Ncal, npart, ndim):
        nm, th = 100, 0.8  
        if self.inseed == None:
            pass
        else:
            np.random.seed(self.inseed)  
        rolds = np.random.uniform(-1, 1, (npart, ndim))
        psiold = psi(rolds, **psi_args)
        iacc, Nsample = 0, 0
        ERp = np.zeros((Ncal, npart, ndim))
        ERq = np.zeros((Ncal, npart, ndim))
        for itot in range (nm*Ncal):
            rnews = rolds + th*np.random.uniform(-1,1,(npart, ndim))
            psinew = psi(rnews, **psi_args)
            psiratio = (psinew/psiold)**2
            if psiratio > np.random.uniform(0,1):
                rolds = np.copy(rnews)
                psiold = psinew
                iacc +=1
            if (itot%nm)==0:
                s = np.random.binomial(1, 0.5, 1)
                s = np.where(s==0, -1, s)
                ERq[Nsample, :, :] = rolds
                ERp[Nsample, :, :] = self.ERmom(psi, rolds, psi_args, s)
                Nsample += 1
        return rolds, Nsample, ERp, ERq



class quantum_system():

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
        self.q_new, self.p_new = utils.qp_transformation(self.list_of_operations, self.list_of_bs_pair,
                                self.operators, self.q_init, self.p_init, self.N, self.npart, self.ndim)
        return self.hamiltonian(self.q_new, self.p_new, **self.hamiltonian_args)



class utils():

    def pairing_prep(q, p, qp_dimension):
        N, npart, ndim = q.shape
        if qp_dimension == 2:
            temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
            temp_p = tf.reshape(p, shape=(1, N, npart, ndim))
            output = tf.concat((temp_q, temp_p), axis=0)
        elif qp_dimension == 4:
            temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
            temp_p = tf.reshape(p, shape=(1, N, npart, ndim))
            qp_pair = tf.concat((temp_q, temp_p), axis=0)
            qp_pair_temp = tf.reshape(qp_pair, shape=(2, N, npart*ndim))
            qp_pair_list = []
            for i in range(npart*ndim):
                qp_pair_list += [qp_pair_temp[:, :, i]]
            output = qp_pair_list
        else:
            sys.exit("The qp_dimension argument must be either 2 or 4.")
        return output

    def reverse_pairing_prep(qp_pair, qp_dimension, N, npart, ndim):
        if qp_dimension == 2:
            q = qp_pair[0, :, :, :]
            p = qp_pair[1, :, :, :]  
        elif qp_dimension == 4:
            output = qp_pair.copy()
            for i in range(len(output)):
                output[i] = tf.reshape(output[i], shape=(2, N, 1))
            output = tf.concat((output), axis=2)
            output = tf.reshape(output, shape=(2, N, npart, ndim))
            q = output[0, :, :, :]
            p = output[1, :, :, :]
        else:
            sys.exit("The qp_dimension argument must be either 2 or 4.")
        return q, p

    def operate_2(operators, qp_pair, is_shift=False):
        #
        _, N, npart, ndim = qp_pair.shape
        #
        output = []
        #
        for i in range(npart):
            output_part = []
            for j in range(ndim):
                if is_shift:
                    output_part += [tf.reshape(tf.math.add(qp_pair[:, :, i, j], operators[i][j]), shape=(2, N, 1, 1))]
                else:
                    output_part += [tf.reshape(operators[i][j] @ qp_pair[:, :, i, j], shape=(2, N, 1, 1))]
            output_part = tf.concat(output_part, axis=3)
            output += [output_part]
        #
        qp_pair = tf.concat(output, axis=2)
        return qp_pair


    def operate_4(operators, qp_pair, bs_pair):
        output = qp_pair.copy()
        for i in range(len(bs_pair)):
            first_pair = qp_pair[bs_pair[i][0]]
            second_pair = qp_pair[bs_pair[i][1]]
            #
            temp = tf.concat((first_pair, second_pair), axis=0)
            temp = operators[i] @ temp
            #
            output[bs_pair[i][0]] = temp[:2, :]
            output[bs_pair[i][1]] = temp[2:, :]
        qp_pair = output
        return qp_pair
  

    def initialize_gates(list_of_operations, list_of_bs_pair, variables, npart, ndim=3):
        operators = []
        var_position = 0
        bs_gates_id = 0
        for i in range(len(list_of_operations)):
            operation = list_of_operations[i]
            if operation == "squeeze":
                op = gates.squeeze_gate(variables[var_position:var_position+npart*ndim], npart, ndim)
                operators += [op]
                var_position = var_position+npart*ndim
            elif operation == "shift":
                op = gates.shift_gate(variables[var_position:var_position+2*npart*ndim], npart, ndim)
                operators += [op]
                var_position = var_position+2*npart*ndim
            elif operation == "rotate":
                op = gates.rotate_gate(variables[var_position:var_position+npart*ndim], npart, ndim)
                operators += [op]
                var_position = var_position+npart*ndim
            elif operation == "beam_splitter":
                bs_pair = list_of_bs_pair[bs_gates_id]
                op = gates.beam_splitter_gate(variables[var_position:var_position+len(bs_pair)], bs_pair)
                operators += [op]
                var_position = var_position+len(bs_pair)
                bs_gates_id += 1
            elif operation == "single_cubic":
                op = variables[var_position:var_position+npart*ndim]
                operators += [op]
                var_position = var_position+npart*ndim
            elif operation[:18] == "fixed_single_cubic":
                operators += [float(operation[19:])]
            elif operation == "sigmoid":
                operators += ["sigmoid"]
            elif operation == "tanh":
                operators += ["tanh"]
            elif operation == "trainable_sigmoid":
                op = variables[var_position:var_position+npart*ndim]
                operators += [op]
                var_position = var_position+npart*ndim
            elif operation == "trainable_tanh":
                op = variables[var_position:var_position+npart*ndim]
                operators += [op]
                var_position = var_position+npart*ndim

        return operators

    def qp_transformation(list_of_operations, list_of_bs_pair, operators, q, p, N, npart, ndim=3):
        operation_dimension = {
            "squeeze": 2,
            "shift": 2,
            "rotate": 2,
            "beam_splitter": 4,
            "single_cubic": 2,
            "fixed_single_cubic": 2,
            "sigmoid": 2,
            "tanh": 2,
            "trainable_sigmoid": 2,
            "trainable_tanh": 2
        }
        qp_dimension = 2
        qp_pair = utils.pairing_prep(q, p, qp_dimension)  # create the new pair
        bs_gates_id = 0
        list_of_variables = []
        for i in range(len(list_of_operations)):
            operation = list_of_operations[i]
            if operation[:18] == "fixed_single_cubic":
                operation = "fixed_single_cubic"
            op = operators[i]
            # prepare the qp pair
            if operation_dimension.get(operation, "Error!") == "Error!":
                sys.exit('The list of operations must only contains "squeeze", "shift", "rotate", "beam_splitter", and/or a name of non-linear activation function available.')
            if operation_dimension.get(operation, "Error!") == qp_dimension:
                pass
            else:
                q, p = utils.reverse_pairing_prep(qp_pair, qp_dimension, N, npart, ndim) # reset the q and p to the initial shape
                qp_pair = utils.pairing_prep(q, p, operation_dimension.get(operation, "Error!"))  # create the new pair
                qp_dimension = operation_dimension.get(operation, "Error!")

            # do the operation on the qp pair
            if operation == "squeeze" or operation == "rotate":
                qp_pair = utils.operate_2(op, qp_pair, is_shift=False)
            elif operation == "shift":
                qp_pair = utils.operate_2(op, qp_pair, is_shift=True)
            elif operation == "beam_splitter":
                qp_pair = utils.operate_4(op, qp_pair, list_of_bs_pair[bs_gates_id])
                bs_gates_id += 1
            elif operation == "single_cubic":
                qp_pair = nonlin_activation.single_mode_cubic_phase_gate(op, qp_pair)
            elif operation == "fixed_single_cubic":
                qp_pair = nonlin_activation.fixed_single_mode_cubic_phase_gate(float(list_of_operations[i][19:]), qp_pair)
            elif operation == "sigmoid":
                qp_pair = nonlin_activation.sigmoid(qp_pair)
            elif operation == "tanh":
                qp_pair = nonlin_activation.tanh(qp_pair)
            elif operation == "trainable_sigmoid":
                qp_pair = nonlin_activation.trainable_sigmoid(op, qp_pair)
            elif operation == "trainable_tanh":
                qp_pair = nonlin_activation.trainable_tanh(op, qp_pair)

        # after all the operations have been done
        q, p = utils.reverse_pairing_prep(qp_pair, qp_dimension, N, npart, ndim) # reset the q and p to the initial shape

        return q, p

    def calculate_variables(list_of_operations, list_of_bs_pair, npart, ndim=3):
        squeeze = list_of_operations.count("squeeze")
        shift = list_of_operations.count("shift")
        rotate = list_of_operations.count("rotate")
        beam_splitter = list_of_operations.count("beam_splitter")
        trainable_activation = list_of_operations.count("single_cubic") + list_of_operations.count("trainable_sigmoid") + list_of_operations.count("trainable_tanh")
        total_bs_pair = 0
        if beam_splitter != 0:
            for i in range(len(list_of_bs_pair)):
                total_bs_pair += len(list_of_bs_pair[i])

        total_variables = squeeze*npart*ndim + shift*2*npart*ndim + \
            rotate*npart*ndim + total_bs_pair + trainable_activation*npart*ndim

        return total_variables

    def xyz_to_numbers(pair_list, ndim):
        temp = copy.deepcopy(pair_list)
        for i in range(len(temp)):
            for j in range(len(temp[0])):
                for k in range(len(temp[0][0])):
                    temp[i][j][k] = ndim*(int(pair_list[i][j][k][0]) - 1) + ord(pair_list[i][j][k][1]) - ord('x')
        return temp


class gates():

    def squeeze_gate(variables, npart, ndim=3):
        operators = []
        for i in range(npart):
          operator_part = []
          for j in range(ndim):
            operator_part += [tf.linalg.diag([1/variables[i*ndim + j], variables[i*ndim + j]])]
          operators += [operator_part]
        return operators

    def shift_gate(variables, npart, ndim=3):
        variables = tf.reshape(variables, shape=(2, npart*ndim))
        operators = []
        for i in range(npart):
            operator_part = []
            for j in range(ndim):
                operator_part += [tf.reshape(variables[:, i*ndim + j], shape=(2,1))]
            operators += [operator_part]
        return operators

    def rotate_gate(variables, npart, ndim=3):
        operators = []
        for i in range(npart):
            operator_part = []
            for j in range(ndim):
                operator_part += [
                    tf.linalg.diag([tf.math.cos(variables[i*ndim + j]),
                    tf.math.cos(variables[i*ndim + j])]) +
                    tf.reverse(tf.linalg.diag([-tf.math.sin(variables[i*ndim + j]),
                    tf.math.sin(variables[i*ndim + j])]), axis=[0])
                ]
            operators += [operator_part]
        return operators

    def beam_splitter_gate(variables, bs_pair):
        operators = []
        for i in range(len(bs_pair)):
            bs_corner = tf.linalg.diag([tf.math.cos(variables[i]), tf.math.cos(variables[i])]) + tf.reverse(tf.linalg.diag([tf.math.sin(variables[i]), -tf.math.sin(variables[i])]), axis=[0])
            temp_top = tf.concat((bs_corner, tf.zeros((2,2), dtype='double')), axis=1)
            temp_bottom = tf.concat((tf.zeros((2,2), dtype='double'), bs_corner), axis=1)
            bs_gate = tf.concat((temp_top, temp_bottom), axis=0)
            operators += [bs_gate]
        #
        return operators



class nonlin_activation():

    def single_mode_cubic_phase_gate(var, qp_pair):
        q = qp_pair[0, :, :, :]
        p = qp_pair[1, :, :, :]
        N, npart, ndim = q.shape
        var = tf.reshape(var, shape=(1, npart, ndim))
        p_new = p + var*(q**2)
        temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
        temp_p = tf.reshape(p_new, shape=(1, N, npart, ndim))
        return tf.concat((temp_q, temp_p), axis=0)

    def fixed_single_mode_cubic_phase_gate(constant_parameter, qp_pair):
        q = qp_pair[0, :, :, :]
        p = qp_pair[1, :, :, :]
        N, npart, ndim = q.shape
        p_new = p + constant_parameter*(q**2)
        temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
        temp_p = tf.reshape(p_new, shape=(1, N, npart, ndim))
        return tf.concat((temp_q, temp_p), axis=0)

    def sigmoid(qp_pair):
        q = qp_pair[0, :, :, :]
        p = qp_pair[1, :, :, :]
        N, npart, ndim = q.shape
        p_new = p + tf.math.sigmoid(q)
        temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
        temp_p = tf.reshape(p_new, shape=(1, N, npart, ndim))
        return tf.concat((temp_q, temp_p), axis=0)
  
    def tanh(qp_pair):
        q = qp_pair[0, :, :, :]
        p = qp_pair[1, :, :, :]
        N, npart, ndim = q.shape
        p_new = p + tf.math.tanh(q)
        temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
        temp_p = tf.reshape(p_new, shape=(1, N, npart, ndim))
        return tf.concat((temp_q, temp_p), axis=0)

    def trainable_sigmoid(var, qp_pair):
        q = qp_pair[0, :, :, :]
        p = qp_pair[1, :, :, :]
        N, npart, ndim = q.shape
        var = tf.reshape(var, shape=(1, npart, ndim))
        p_new = p + tf.math.sigmoid(var*q)
        temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
        temp_p = tf.reshape(p_new, shape=(1, N, npart, ndim))
        return tf.concat((temp_q, temp_p), axis=0)

    def trainable_tanh(var, qp_pair):
        q = qp_pair[0, :, :, :]
        p = qp_pair[1, :, :, :]
        N, npart, ndim = q.shape
        var = tf.reshape(var, shape=(1, npart, ndim))
        p_new = p + tf.math.tanh(var*q)
        temp_q = tf.reshape(q, shape=(1, N, npart, ndim))
        temp_p = tf.reshape(p_new, shape=(1, N, npart, ndim))
        return tf.concat((temp_q, temp_p), axis=0)


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
    oms=np.array([1., 1., 1.]),
    deltaq=0.5,
    aa=1.0
)
# deltaq: terkait shifting gate (Killoran)


sampler = monte_carlo(inseed=8735)

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
    two_body_interaction = 0
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

