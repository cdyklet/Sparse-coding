import numpy as np
import matplotlib.pyplot as plt


def plot_rf(rf, out_dim, M):
    rf = rf.reshape(out_dim, -1)
    # normalization
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(out_dim, M, M)
    # plotting
    n = int((np.sqrt(rf.shape[0]))//1)
    # n = min([n, 10])
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize = (20, 20))
    for i in range(n**2):
        ax = axes[i // n][i % n]
        ax.imshow(rf[i], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    # for j in range(rf.shape[0], n * n):
    #     ax = axes[j // n][j % n]
    #     ax.imshow(np.ones_like(rf[0]) * -1, cmap="grey", vmin=-1, vmax=1)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_aspect("equal")
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig
