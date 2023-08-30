import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from .utils import get_lower_triangular_from_diag, get_lower_triangular_uniform_random
    
class RBFAID(gpflow.kernels.Kernel):  

    def __init__(self, **kwargs):
        randomized = kwargs["randomized"]
        self.D = kwargs["D"]
        self.input_dim = self.D
        self.T = kwargs["T"]
        self._v = kwargs["variance"]
        self.kernel_computation_type = kwargs["kernel_computation_type"]
        if not randomized:
            L = get_lower_triangular_from_diag(self.D)
        else:
            L = get_lower_triangular_uniform_random(self.D)
        super().__init__()
        self.L = tf.Variable(L, name='L', dtype=tf.float64) # D*(D+1)/2
        self.logvariance = tf.Variable(np.log(self._v), dtype=tf.float64, name='log_variance', trainable=False)
        self.variance = tf.exp(self.logvariance)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        X1 = X
        N1 = tf.shape(X1)[0]
        N2 = tf.shape(X2)[0]
        # Reshape X1, X2
        X1 = tf.reshape(X1, (N1, self.T, self.D)) # N1 x T x D
        X2 = tf.reshape(X2, (N2, self.T, self.D)) # N2 x T x D
        Lambda = self.precision() # D x D
        z = tf.reduce_sum(tf.multiply(tf.matmul(X1,Lambda), X1), axis=2) # N1 x T -> z[sample]->[k(y0,y0),k(y1,y1),...,k(yT,yT)]
        z_long  = tf.tile(tf.expand_dims(z, 2), [1, 1, self.T]) # N1 x T x T
        X11r = tf.tile(tf.expand_dims(z_long, 3), [1, 1, 1, N2]) # N1 x T x T x N2
        #---
        z = tf.reduce_sum(tf.multiply(tf.matmul(X2,Lambda), X2), axis=2) # N2 x T
        z_long  = tf.tile(tf.expand_dims(z, 2), [1, 1, self.T])
        X22r = tf.tile(tf.expand_dims(z_long, 3), [1, 1, 1, N1]) 
        X22r = tf.transpose(X22r, perm=[3,2,1,0])
        X12 = tf.einsum('ijk,kk,lpk->ijpl',X1,Lambda,X2) # N1 x T x T x N2
        K = X11r + X22r - 2*X12
        Kxx =  self.variance * tf.math.exp(-0.5 * K) # N1 x T x T x N2
        if self.kernel_computation_type == 'single_sum':
            mask = tf.broadcast_to(tf.expand_dims(tf.expand_dims(tf.eye(self.T,  dtype=tf.float64), 0), -1), (N1, self.T, self.T, N2))
            Kxx = tf.multiply(Kxx, mask)
        return tf.reduce_sum(Kxx, axis=(1,2))
    
    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X,X)) 
    
    def precision(self):
        L = tfp.math.fill_triangular(self.L, upper=False)
        Lambda = tf.linalg.matmul(L, tf.transpose(L))
        return Lambda
    
    def precision_off_diagonals(self):
        diag_L = tf.linalg.tensor_diag_part(self.precision())
        return self.precision() - tf.linalg.diag(diag_L)
    
    def precision_off_diagonals_prot(self):
        return tf.boolean_mask(self.precision(), ~tf.eye(self.d, self.d, dtype=tf.bool))