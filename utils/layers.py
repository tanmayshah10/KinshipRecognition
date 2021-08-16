import tensorflow as tf
from tensorflow.keras.layers import Layer


class RBF(Layer):
    
    '''Returns kernel dot product for RBF kernel. The scale (gamma) parameter is trainable.'''
    
    def __init__(self, **kwargs):
        super(RBF, self).__init__(**kwargs)

    def build(self, input_shape):
        # Note: The scale parameter in this implementation is of the same size as input dims.
        self.gamma = self.add_weight(shape=(int(input_shape[1])),
                                     initializer='ones',
                                     trainable=True)
        super(RBF, self).build(input_shape)

    def call(self, x1, x2):
        # Squared L2
        l2 = tf.math.pow(x1 - x2, 2)
        return tf.math.exp(-tf.math.multiply(l2, self.gamma))


class CrossAttention(Layer):
    
    '''Computes cross attention between 2 inputs.'''
    
    def __init__(self, units, kernel_dim2d, value_dim, qk_dim=None, batch_size=16,
                 kernel_initializer="glorot_uniform", kernel_regularizer='l2', **kwargs):
    
        super(CrossAttention, self).__init__(**kwargs)
        
        # Number of output channels (attention heads)
        self.units = units
        # Dimension of input. In the case of images it is (H x W).
        self.value_dim = value_dim
        assert len(kernel_dim2d) == 2
        # Dimensions of output image.
        self.kernel_dim2d = kernel_dim2d
        self.kernel_len = kernel_dim2d[0] * kernel_dim2d[1]
        # Dimension of query and key weights.
        self.qk_dim = qk_dim if qk_dim else kernel_len
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        
        self.N = batch_size
        self.key_w = None
        self.query_w = None
        self.value_w = None
        # Scaling Q*K dot product for stability.
        self.scale = self.value_dim ** -0.5
        
    def build(self, input_shape):
        
        self.query_w = self.add_weight(shape=(self.units, self.value_dim, self.qk_dim),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       trainable=True, name='query_w')
        self.key_w = self.add_weight(shape=(self.units, self.value_dim, self.qk_dim),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True, name='key_w')
        self.value_w = self.add_weight(shape=(self.units, self.value_dim, self.kernel_len),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True, name='value_w')

    def call(self, query, value, key=None):    
        
        # Query, values, and keys are of shape=(N, h, w, c)
        if not key:
            key = value
        
        _, h, w, c = value.shape
        query = tf.transpose(query, perm=(0, 3, 1, 2))
        query = tf.reshape(query, shape=(self.N, c, h*w))
        key = tf.transpose(key, perm=(0, 3, 1, 2))
        key = tf.reshape(key, shape=(self.N, c, h*w))
        value = tf.transpose(value, perm=(0, 3, 1, 2))
        value = tf.reshape(value, shape=(self.N, c, h*w))
        
        qW = tf.matmul(tf.expand_dims(query, 1), self.query_w)
        kW = tf.matmul(tf.expand_dims(key, 1), self.key_w)
        dot = tf.matmul(qW, tf.transpose(kW, perm=(0, 1, 3, 2,)))
        attn_w = tf.nn.softmax(dot * self.scale)
        vW = tf.matmul(tf.expand_dims(value, 1), self.value_w)
        
        flat_kernel = tf.einsum('ijkl, ijlm -> ijkm', attn_w, vW)
        kernel = tf.reshape(flat_kernel, shape=(self.N, self.units, c, self.kernel_dim2d[0], self.kernel_dim2d[1]))
        kernel = tf.transpose(kernel, perm=(0, 3, 4, 2, 1))
        return kernel
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({'units': self.units,
                       'kernel_dim2d': self.kernel_dim2d,
                       'value_dim': self.value_dim,
                       'qk_dim': self.qk_dim})
        return config    
