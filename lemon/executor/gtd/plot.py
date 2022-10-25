"""
Helper functions for plotting
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from gtd.io import makedirs
from gtd.log import in_ipython


def hinton(matrix, max_weight=None, ax=None, xtick=None, ytick=None, inverted_color=False):
    """Draw Hinton diagram for visualizing a weight matrix.

    Copied from: http://matplotlib.org/examples/specialty_plots/hinton_demo.html
    """
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        if inverted_color:
            color = 'black' if w > 0 else 'white'
        else:
            color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w))
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

    if xtick:
        ax.set_xticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels(xtick)
    if ytick:
        ax.set_yticks(np.arange(matrix.shape[1]))
        ax.set_yticklabels(ytick)
    return ax


def show(title, directory=''):
    """If in IPython, show, otherwise, save to file."""
    import matplotlib.pyplot as plt
    if in_ipython():
        plt.show()
    else:
        # ensure directory exists
        makedirs(directory)

        plt.savefig(os.path.join(directory, title) + '.png')
        # close all figures to conserve memory
        plt.close('all')


def plot_pdf(x, cov_factor=None, *args, **kwargs):
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    density = gaussian_kde(x)
    xgrid = np.linspace(min(x), max(x), 200)
    if cov_factor is not None:
        density.covariance_factor = lambda: cov_factor
        density._compute_covariance()
    y = density(xgrid)
    plt.plot(xgrid, y, *args, **kwargs)


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb