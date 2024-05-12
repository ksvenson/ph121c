import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lab1 import main as l1main
import utility

CACHE_DIR = './data/'
FIGS_DIR = './figs/'

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}

LSPACE = np.arange(2, 3)

FIELD_VALS = {'hx': 1, 'hz': 0}


# @utility.cache('npy', CACHE_DIR + 'dense_H')
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

        # for flip in range(0, L):
        #     H[i ^ (1 << flip), i] -= FIELD_VALS['hx']
    return H


# @utility.cache('npz', CACHE_DIR + 'l3_dense_eigs')
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


def p4_1_debug():
    for L in LSPACE:
        ham = make_dense_H(L)
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']

        # test_ham = np.load(f'../lab2/data/dense_H_L{L}.npz')['loop'][0]
        # test_eigs = np.load(f'../lab2/data/dense_eigs_loop_L{L}.npz')
        # test_evals = test_eigs['evals'][0]

        test_ham = l1main.make_dense_H(L, hspace=[FIELD_VALS['hx']], note=f'DELETE_ME_L{L}_hx{FIELD_VALS["hx"]}')['loop']
        test_eigs = l1main.dense_eigs(test_ham, hspace=[FIELD_VALS['hx']], note=f'DELETE_ME_L{L}_hx{FIELD_VALS["hx"]}')
        test_evals = test_eigs['evals'][0]

        evals = np.sort(evals)
        test_evals = np.sort(test_evals)

        print(f'ham:\n{ham}')
        print(f'test_ham:\n{test_ham[0]}')

        print(f'eval difference: {np.linalg.norm(evals - test_evals)}')
        print(f'ham difference: {np.linalg.norm(ham - test_ham[0])}')


def p4_1():
    # Pauli matrices defined in a basis where first component is spin down, second component is spin up.
    ops = {
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, 1j], [-1j, 0]]),
        'z': np.array([[-1, 0], [0, 1]])
    }
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    for L in LSPACE:
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']
        t_space = np.linspace(0, 10, 100)
        xi_state = make_xi_state(L)
        # convert xi state to diagonal basis
        xi_state = evecs.T.conj() @ xi_state
        # c_m^* c_n
        coeffs = np.multiply.outer(xi_state.conj(), xi_state)
        # (e_n - e_m) t
        eng_diff = np.add.outer(-1 * evals, evals)
        propagator = np.exp(-1j * np.multiply.outer(t_space, eng_diff))
        for op_idx, op in enumerate(ops):
            og_xi_state = make_xi_state(L)
            global_op = np.kron(ops[op], np.identity(2**(L-1)))
            print(f'{op}0: {np.sum(og_xi_state.conj() * (global_op @ og_xi_state))}')

            Omn = rebase_operator(L, ops[op], evecs)
            # Omn = rebase_operator(L, np.identity(2), evecs)

            # Omn = evecs.T.conj() @ np.identity(2**L) @ evecs

            # Omn = np.diag(-2 * (np.arange(2**L) // 2**(L-1)) + 1)
            # Omn = evecs.T.conj() @ np.diag(-2 * (np.arange(2**L) % 2) + 1) @ evecs

            O_evolution = np.sum(coeffs * propagator * Omn, axis=(1, 2))

            axes[op_idx].plot(t_space, O_evolution, label=f'L={L}')
            axes[op_idx].set_xlabel('Time $t$')
            axes[op_idx].set_title(rf'$\mu={op}$')

    axes[0].set_ylabel(r'$\langle \sigma_1^\mu(t) \rangle$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'p4_1_1.png')

def p4_1_test():
    # Pauli matrices defined in a basis where first component is spin down, second component is spin up.
    ops = {
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, 1j], [-1j, 0]]),
        'z': np.array([[-1, 0], [0, 1]])
    }
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    for L in LSPACE:
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']
        t_space = np.linspace(0, 10, 100)
        xi_state = make_xi_state(L)
        # convert xi state to diagonal basis
        xi_state = evecs.T.conj() @ xi_state
        for op_idx, op in enumerate(ops):
            data = []
            Omn = rebase_operator(L, ops[op], evecs)
            for t in t_space:
                evolved = np.exp(-1j * evals * t) * xi_state
                data.append(np.sum(evolved.conj() * (Omn @ evolved)))

            axes[op_idx].plot(t_space, data, label=f'L={L}')
            axes[op_idx].set_xlabel('Time $t$ testing')
            axes[op_idx].set_title(rf'$\mu={op}$')

    axes[0].set_ylabel(r'$\langle \sigma_1^\mu(t) \rangle$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'testing_p4_1_1.png')


if __name__ == '__main__':
    # p4_1()
    # p4_1_test()
    p4_1_debug()
