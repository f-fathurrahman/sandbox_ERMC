import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

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