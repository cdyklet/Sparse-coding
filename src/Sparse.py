import tensorflow as tf
import numpy as np

sigma = 1
tf.random.set_seed(1)


def log_func(x):
    x = x / sigma
    return tf.math.reduce_sum(tf.math.log(1 + tf.square(x)))


def abs_func(x):
    x = x / sigma
    return tf.math.reduce_sum(tf.math.abs(x))


class SparseNet(tf.keras.Model):
    def __init__(self, K: int, M: int, R_lr: float = 0.1, lmda: float = 5e-3, optimizer=tf.keras.optimizers.SGD()):
        super(SparseNet, self).__init__()
        self.K = K
        self.M = M
        self.R_lr = R_lr
        self.lmda = lmda
        self.U = tf.keras.layers.Dense(self.M ** 2, use_bias=False)
        self.optimizer = optimizer
        self.R = None
        self.converged = tf.Variable(False, dtype=tf.bool)
        self.old_loss = tf.Variable(1, dtype= tf.float32)

    def threshold(self, x, alpha):
        return tf.nn.relu(x - alpha) - tf.nn.relu(-x - alpha)
        #return tf.maximum((tf.math.abs(x) - alpha), 0) * tf.math.sign(x) 

    def soft_thresholdind(self,):
        self.R.assign(self.threshold(self.R, self.lmda))

    def zero_grad(self):
        pass

    def normalize_weights(self):
        weights = self.U.weights[0]
        norm_val = tf.norm(weights, ord=2, axis=1)
        weights = tf.transpose(weights) / norm_val
        weights = tf.transpose(weights)
        self.U.weights[0].assign(weights)

    def build(self, input_shape=None):
        self.R = tf.Variable(lambda: tf.random.normal((input_shape[0], self.K), dtype=tf.float32))
        self.U.build(input_shape=(self.K,))
        super(SparseNet, self).build(input_shape)
        self.weight_initializer()
        self.normalize_weights()
        self.built = True
        
         
    def weight_initializer(self,):
        stdv = tf.math.sqrt(tf.cast(self.K, dtype=tf.float32))
        weights = tf.random.uniform(shape=(self.K, self.M ** 2), minval=-stdv, maxval=stdv)
        cov = tf.matmul(weights, tf.transpose(weights))
        diag = tf.linalg.diag_part(cov)
        weights = tf.transpose(weights) / diag
        weights = tf.transpose(weights)
        self.U.weights[0].assign(weights)

    def call(self, img_batch, training= False):
        if training:
            self.ista_(img_batch)
        # now predict again
        pred = self.U(self.R)
        return pred

    def sparse_loss(self, func=abs_func):
        return func(self.R)

    def loss(self, image):
        # pred
        pred = self.U(self.R)
        # preserve loss
        loss = tf.math.reduce_sum(tf.square(image - pred))
        # sparseness loss
        # spars = self.sparse_loss()
        # loss = loss + self.lmda * spars
        return loss
    @tf.function
    def ista_(self, img_batch):
        # ista 
        self.R.assign(tf.random.normal(self.R.shape))
        self.converged.assign(False)
        self.old_loss.assign(float('inf'))
        eye = tf.eye(20000)
        while not self.converged:
            pred = self.U(self.R)
            error = pred - img_batch
            loss = tf.math.reduce_sum(tf.math.square(error)) + tf.math.reduce_sum(tf.math.abs(self.R))  
            gradient = 2 * tf.matmul(tf.matmul(eye, error), tf.transpose(self.U.weights[0]))
            self.R.assign(self.R - self.R_lr* gradient)
            self.soft_thresholdind()
            c = tf.math.logical_or(tf.abs(loss - self.old_loss) < 0.2, loss > self.old_loss)
            # tf.print(loss, tf.math.count_nonzero(self.R), self.old_loss,)
            self.converged.assign(c)
            self.old_loss.assign(loss)
