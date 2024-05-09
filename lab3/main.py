import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lab1 import main as l1main
import utility

CACHE_DIR = './data/'


@utility.cache('npz', CACHE_DIR + 'dense_H')
def make_dense_H(L, hx=-1.05, hz=0.5, note=None):
    """
    Makes Hamiltonian in sigma_z basis as defined in Equation 8 with periodic boundary conditions.
    States are indexed in binary: 0 for spin down and 1 for spin up.
    :param L: system size.
    :param note: appeneded to file name when caching function output.
    :return: Hamiltonian with shape (2**L, 2**L).
    """
    states = np.arange(2**L)
    # Perform a XOR between states, and states shifted by 1 bit. Make sure that last bit is set to zero.
    bulk = 2 * ((states & ~(1 << L-1)) ^ (states >> 1)).bit_count() - (L-1)
    # Include the periodic term.
    loop = 2 * ((states ^ (states >> (L - 1))) % 2) - 1
    # h_z field
    z_field = 2 * states.bit_count() - 1
    H = np.diag(bulk + loop)
    for i in range(dim):
        H_open[i, i] = 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)
        for flip in range(0, L):
            H_open[:, i ^ (1 << flip), i] -= hspace
        H_loop[:, :, i] = H_open[:, :, i]
        H_loop[:, i, i] += 2 * ((i ^ (i >> (L - 1))) % 2) - 1
    return {'open': H_open, 'loop': H_loop}


def p4_1():
    pass


if __name__ == '__main__':
    pass
