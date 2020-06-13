import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

sys.path.insert(0, os.path.abspath("../"))
from tqdm import tqdm
from Sparse import SparseNet
from tensorflow.data import Dataset
import tensorflow
from ImageDataset import NatPatchDataset, load_data
from utils import parse_args
from plotting import plot_rf
import matplotlib.pyplot as plt


print(tf.__version__)

gpu = tf.config.experimental.get_visible_devices("GPU")[0]
tf.config.experimental.set_memory_growth(gpu, enable=True)
tb_path = "../output/m1"

tb_path = "../output/m1"


class test:
    batch_size = 2000
    n_neuron = 400
    size = 10
    epoch = 100
    learning_rate = 1e-2
    r_learning_rate = 5e-3
    reg = 5e-3


arg = test()
dataloader = tf.data.Dataset.from_tensor_slices(load_data(arg.batch_size, arg.size, arg.size))
print("loading data finished")
tf.autograph.set_verbosity(0)
batch_size = 20000


@tf.function
def train(epochs, sparse_net, optimizer_, running_loss, data):
    tf.print('begin')
    for e in range(epochs):
        running_loss.assign(0)
        for img_batch in data.batch(batch_size):
            img_batch = tf.keras.layers.Flatten()(img_batch)
            with tf.GradientTape() as tape:
                tape.watch(sparse_net.U.weights[0])
                pred = sparse_net(img_batch, training=True)
                loss = tf.math.reduce_sum(tf.square(img_batch - pred))
            running_loss.assign_add(loss)
            gradients = tape.gradient(loss, sparse_net.U.weights[0])
            _ = optimizer_.apply_gradients([(gradients, sparse_net.U.weights[0])])
        _ = sparse_net.normalize_weights()
        if e % 1 == 0:
            persever_loss = running_loss
            total_loss = persever_loss + tf.math.reduce_sum(tf.math.abs(sparse_net.R))
            sparse_loss = total_loss - persever_loss
            nonzero_number = tf.math.count_nonzero(sparse_net.R)
            max_val = tf.math.reduce_max(sparse_net.weights[0])
            weight = tf.abs(sparse_net.weights[0])
            zero_mask = tf.where(weight > 0, weight, float("inf"))
            min_val = tf.math.reduce_min(zero_mask)
            tf.print(
                "Iter:",
                e,
                ",Total loss: ",
                total_loss,
                "persever loss:",
                persever_loss,
                ", sparse loss:",
                sparse_loss,
                " nonzero number:",
                nonzero_number,
                ", max val:",
                max_val,
                "min val:",
                min_val,
            )
    tf.print("finished.")
    return


sparse_net = SparseNet(K=arg.n_neuron, M=arg.size, R_lr=arg.r_learning_rate, lmda=arg.reg, optimizer=None)
sparse_net.build(input_shape=(batch_size,))
print('model building finished')
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
running_loss_ = tf.Variable(0, dtype=tf.float32)
tf.config.experimental_run_functions_eagerly(True)
train(2000, sparse_net, optimizer, running_loss_, dataloader)
