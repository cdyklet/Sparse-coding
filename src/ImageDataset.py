import tensorflow as tf
from scipy.io import loadmat
import numpy as np
# from tensorflow.data import Dataset


class NatPatchDataset:
    def __init__(self, N, width, height, border=4, fpath="./data/IMAGES.mat"):
        self.N = N
        self.width = width
        self.height = height
        self.border = border
        self.fpath = fpath
        self.extract_patches()

    def __getitem__(self, idx=0):
        return self.images[idx]

    def __len__(self):
        return self.images.shape[0]

    def extract_patches(self):
        # load mat
        X = loadmat(self.fpath)
        X = X["IMAGES"].astype(np.float32)
        image_size = X.shape[0]
        n_img = X.shape[2]
        self.images = tf.Variable(tf.zeros((self.N * n_img, self.width, self.height)), dtype=tf.float32)
        # for every images
        counter = 0
        for i in range(n_img):
            img = X[:, :, i]
            for j in range(self.N):
                x = np.random.randint(self.border, image_size - self.width - self.border)
                y = np.random.randint(self.border, image_size - self.height - self.border)
                crop = img[x : x + self.width, y : y + self.height]
                crop_mean = crop - tf.math.reduce_mean(crop)
                self.images[counter, ...].assign(crop_mean)
                counter += 1


def load_data(N, width, height, border=4, fpath="../data/IMAGES.mat"):
    X = loadmat(fpath)
    X = tf.constant(X["IMAGES"].astype(np.float32))
    image_size = X.shape[0]
    n_img = X.shape[2]
    images = tf.Variable(tf.zeros((N * n_img, width, height)), dtype=tf.float32)
    # for every images
    counter = 0
    for i in range(n_img):
        img = X[:, :, i]
        for j in range(N):
            x = np.random.randint(border, image_size - width - border)
            y = np.random.randint(border, image_size - height - border)
            crop = img[x : x + width, y : y + height]
            # mean = tf.math.reduce_mean(crop)
            # std = tf.math.reduce_std(crop)
            # crop = (crop - mean) / std
            # # crop = crop - np.mean(crop)
            crop = crop - tf.math.reduce_mean(crop)
            # cov = (crop.T @ crop)/float(X.shape[0])
            # eigVals, eigVec = np.linalg.eig(cov)
            # crop = crop.dot(eigVec)
            images[counter, ...].assign(crop)
            counter += 1
    return images

