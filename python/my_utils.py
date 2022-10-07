import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import copy

from my_gates import gates
from my_nonlin_activation import nonlin_activation

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

