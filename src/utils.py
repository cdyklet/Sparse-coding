import argparse

parser = argparse.ArgumentParser(description="Template")

# model
parser.add_argument("-N", "--batch_size", default=2000, type=int, help="Batch size")
parser.add_argument("-K", "--n_neuron", default=400, type=int, help="The number of neurons")
parser.add_argument("-M", "--size", default=10, type=int, help="The size of receptive field")
parser.add_argument("-e", "--epoch", default=100, type=int, help="Number of Epochs")
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="Learning Rate")
parser.add_argument("-rlr", "--r_learning_rate", default=1e-3, type=float, help="Learning rate for ISTa")
parser.add_argument("-lmda", "--reg", default=5e-3, type=float, help="LSTM hidden size")
# parser arguments
def parse_args():
    return parser.parse_args()

