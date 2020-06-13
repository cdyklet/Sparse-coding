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
optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-1)
sparse_net = SparseNet(K=arg.n_neuron, M=arg.size, R_lr=arg.r_learning_rate, lmda=arg.reg, optimizer=optimizer1)
sparse_net.build(input_shape=(batch_size,))
num = tf.data.experimental.cardinality(dataloader).numpy()
optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-3)
optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-3)
print("model build finished")
min_loss = tf.Variable(float('inf'), dtype = tf.float32)
cur_iter = tf.Variable(0, dtype = tf.int32)
old_loss = tf.Variable(0, dtype = tf.float32)
converged = tf.Variable(False, dtype = tf.bool)
for e in range(1000):
    c = 0
    i = 0
    min_loss.assign(float('inf'))
    cur_iter.assign(0)
    old_loss.assign(0)
    converged.assign(False)
    # _ = dataloader.shuffle(num, reshuffle_each_iteration = True)
    # while not converged:
    running_loss = 0
    for img_batch in dataloader.batch(batch_size):
        img_batch = tf.reshape(img_batch, (img_batch.shape[0], -1))
        with tf.GradientTape() as tape:
            tape.watch(sparse_net.U.weights[0])
            pred = sparse_net(img_batch, training= True)
            loss = tf.math.reduce_sum(tf.square(img_batch - pred))
        running_loss += loss
        gradients = tape.gradient(loss, sparse_net.U.weights[0])
        if e < 1000:
            _ = optimizer1.apply_gradients([(gradients, sparse_net.U.weights[0])])
        else:
            _ = optimizer2.apply_gradients([(gradients, sparse_net.U.weights[0])])
        
        # if loss < min_loss:
        #     min_loss.assign(loss)
        #     cur_iter.assign(0)
        # else:
        #     cur_iter.assign_add(1)
        # i += 1
        # c = tf.math.logical_or(tf.abs(running_loss - old_loss) < 0.1, running_loss > old_loss)
        # # converged.assign(tf.abs((old_loss - running_loss)/old_loss) < 0.0001)
        # converged.assign(c)ver loss: {:.2f}, sparse loss: {:.2f}, nonzero number {}, max val:{:.2f}, min val:{:.9f}'.format(e,
        # old_loss.assign(running_lover loss: {:.2f}, sparse loss: {:.2f}, nonzero number {}, max val:{:.2f}, min val:{:.9f}'.format(e,ss)
        # if i % 40 == 0:ver loss: {:.2f}, sparse loss: {:.2f}, nonzero number {}, max val:{:.2f}, min val:{:.9f}'.format(e,
        #     print(e, running_loss, tf.math.count_nonzero(sparse_net.R).numpy())
    _ = sparse_net.normalize_weights()
    if e % 1 == 0:
        persever_loss = running_loss.numpy()
        
        total_loss = persever_loss + tf.math.reduce_sum(tf.math.abs(sparse_net.R)).numpy()
        sparse_loss = total_loss - persever_loss
        nonzero_number = tf.math.count_nonzero(sparse_net.R).numpy()
        max_val = tf.math.reduce_max(sparse_net.weights[0])
        weight = tf.abs(sparse_net.weights[0])
        zero_mask = tf.where(weight >0 , weight, float('inf'))
        min_val = tf.math.reduce_min(zero_mask)
        print('Iter: {},Total loss: {:.2f}, persever loss: {:.2f}, sparse loss: {:.2f}, nonzero number {}, max val:{:.2f}, min val:{:.9f}'.format(e, total_loss, persever_loss, sparse_loss, nonzero_number, max_val, min_val))
print("finished.")  # sparse_net = NatPatchDataset(arg.n_neuron, arg.size, R_lr = arg.r_learning_rate, lmda = arg.reg)

weight = sparse_net.U.weights[0]
# weight = tf.transpose(weight)
weight = tf.reshape(weight, (arg.n_neuron, arg.size, arg.size)).numpy()
_ = plot_rf(weight, arg.n_neuron, arg.size)
plt.show()