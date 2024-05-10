import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lab1 import main as l1main
import utility

CACHE_DIR = './data/'

LSPACE = np.arange(8, 11)

FIELD_VALS = {'hx': -1.05, 'hz': 0.5}


@utility.cache('npy', CACHE_DIR + 'dense_H')
def make_dense_H(L, note=None):
    """
    Makes Hamiltonian in sigma_z basis as defined in Equation 8 with periodic boundary conditions.
    States are indexed in binary: 0 for spin down and 1 for spin up.
    :param L: system size.
    :param note: appeneded to filename when caching function output.
    :return: Hamiltonian with shape (2**L, 2**L).
    """
    H = np.zeros((2**L, 2**L))
    for i in range(2**L):
        # Perform a XOR between states, and states shifted by 1 bit. Make sure that last bit is set to zero.
        H[i, i] = 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)
        # Include the periodic term.
        H[i, i] += 2 * ((i ^ (i >> (L - 1))) % 2) - 1
        # h_z field
        H[i, i] += -1 * FIELD_VALS['hz'] * (2 * i.bit_count() - L)
        # h_x field
        flips = np.arange(L)
        H[i ^ (1 << flips), i] += -1 * FIELD_VALS['hx']
    return H


@utility.cache('npz', CACHE_DIR + 'l3_dense_eigs')
def dense_eigs(L, note=None):
    """
    Diagonalizes Hamiltonian from `make_dense_H`.
    :param L: system size.
    :param note: appended to filename when caching function output.
    :return: as a dict: list of evals, matrix with evecs as columns
    """
    print(f'finding dense evals: L={L}')
    H = make_dense_H(L, note=note)
    evals, evecs = sp.linalg.eigh(H)
    return {'evals': evals, 'evecs': evecs}


def make_xi_state(L):
    """
    Makes xi state as defined in equation 9.
    :param L: system size.
    :return: xi state in sigma z basis.
    """
    xi_state = []
    for state in range(2**L):
        num_down = L - state.bit_count()
        xi_state.append((-1*np.sqrt(3))**num_down)
    return np.array(xi_state) / 2**L


def rebase_operator(L, op, evecs):
    """
    Convert the basis of a local 2x2 operator at site 1 from sigma z to the energy basis.
    :param L: system size.
    :param op: 2x2 local operator.
    :return: 2**Lx2**L operator in energy basis
    """
    global_op = np.identity(2**(L-1))
    global_op = np.kron(op, global_op)
    return evecs.T.conj() @ global_op @ evecs


def p4_1():
    for L in LSPACE:
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']
        t_space = np.linspace(0, 10, 1000)
        xi_state = make_xi_state(L)
        # convert xi state to diagonal basis
        xi_state = evecs.T.conj() @ xi_state

        ops = {
            'sx': np.array([[0, 1], [1, 0]]),
            'sy': np.array([[0, -1j], [1j, 0]]),
            'sz': np.array([[1, 0], [0, -1]])
        }

        # use .outer to do stuff

        propagator = np.exp(-1j * np.outer(t_space, evals))
        # states are in rows, each row is propagated a different time
        evolved = propagator * xi_state




if __name__ == '__main__':
    p4_1()