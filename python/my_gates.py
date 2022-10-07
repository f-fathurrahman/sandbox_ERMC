import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

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