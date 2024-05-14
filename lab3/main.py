import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lab2 import main as l2main
import utility

CACHE_DIR = './data/'
FIGS_DIR = './figs/'

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}

LSPACE = np.arange(8, 15, 2)
TSPACE = np.linspace(0, 10, 100)

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
        H[i, i] += 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)
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


def make_prod_state(L, up_coeff, down_coeff):
    """
    Make a translation-invariant product state in the sigma z basis.
    :param L: system size.
    :param up_coeff: spin up coefficient.
    :param down_coeff: spin down coefficient.
    :return: length 2**L array representing the state.
    """
    output = []
    for state in range(2**L):
        num_up = state.bit_count()
        num_down = L - num_up
        output.append((up_coeff**num_up) * (down_coeff**num_down))
    return output


def rebase_operator(L, op, evecs):
    """
    First, take local 2x2 operator `op` and turn it into a global 2**L x 2**L operator in sigma z basis.
    Then, change basis of global operator to energy basis using `evecs`.
    :param L: system size.
    :param op: 2x2 local operator.
    :return: 2**L x 2**L operator in energy basis
    """
    global_op = np.identity(2**(L-1))
    global_op = np.kron(op, global_op)
    return evecs.T.conj() @ global_op @ evecs


def p4_1_12():
    # Pauli matrices defined in a basis where first component is spin down, second component is spin up.
    ops = {
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, 1j], [-1j, 0]]),
        'z': np.array([[-1, 0], [0, 1]])
    }
    fig_sig, axes_sig = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    fig_beta, axes_beta = plt.subplots(figsize=(5, 5))
    for color_idx, L in enumerate(LSPACE):
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']

        xi_state = make_prod_state(L, 1/2, -np.sqrt(3)/2)
        # convert xi state to diagonal basis
        xi_state = evecs.T.conj() @ xi_state

        # c_m^* c_n
        coeffs = np.multiply.outer(xi_state.conj(), xi_state)

        # p4_1_2 stuff
        beta_space = np.linspace(-3, 3, 10000)
        Z_beta = np.array([np.sum(np.exp(-beta * evals)) for beta in beta_space])
        E_beta = np.array([np.sum(np.exp(-beta * evals) * evals) for beta in beta_space]) / Z_beta
        xi_eng = np.sum(xi_state.conj() * evals * xi_state)
        xi_beta_idx = np.argmin(np.abs(E_beta - xi_eng))
        xi_beta = beta_space[xi_beta_idx]

        for op_idx, op in enumerate(ops):
            Omn = rebase_operator(L, ops[op], evecs)
            eng_diff = np.add.outer(-1 * evals, evals)

            O_thermal = np.sum(np.diag(Omn) * np.exp(-1 * xi_beta * evals)) / Z_beta[xi_beta_idx]

            measurement = []
            # Need to loop over `t_space` in order to not run out of memory. Otherwise I would make a dim-3 array
            # for eng_diff to include time.
            for t in TSPACE:
                propagator = np.exp(-1j * eng_diff * t)
                measurement.append(np.sum(coeffs * propagator * Omn))

            axes_sig[op_idx].plot(TSPACE, measurement, label=f'$L={L}$', color=f'C{color_idx}')
            axes_sig[op_idx].axhline(O_thermal, label=rf'$L={L}$ Thermal Limit', color=f'C{color_idx}', linestyle='dotted')
            axes_sig[op_idx].set_xlabel('Time $t$')
            axes_sig[op_idx].set_title(rf'$\mu={op}$')
        axes_beta.plot(beta_space, E_beta, label=rf'$L={L}$')

    axes_sig[0].set_ylabel(r'$\langle \sigma_1^\mu(t) \rangle$')
    handles, labels = axes_sig[0].get_legend_handles_labels()
    fig_sig.legend(handles, labels, **LEGEND_OPTIONS)
    fig_sig.savefig(FIGS_DIR + 'p4_1_1.png', **FIG_SAVE_OPTIONS)

    axes_beta.set_xlabel(r'$\beta$')
    axes_beta.set_ylabel(r'$E_\beta$')
    fig_beta.legend(**LEGEND_OPTIONS)
    fig_beta.savefig(FIGS_DIR + 'p4_1_2_E_beta.png', **FIG_SAVE_OPTIONS)


def p4_1_3():
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    for L in LSPACE:
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']

        xi_state = make_prod_state(L, 1/2, -np.sqrt(3)/2)
        xi_state = evecs.T.conj() @ xi_state

        my_state = make_prod_state(L, -1/np.sqrt(2), 1/np.sqrt(2))
        my_state = evecs.T.conj() @ my_state

        propagator = np.exp(-1j * np.multiply.outer(TSPACE, evals))

        states = {'xi': xi_state, 'my': my_state}
        ax_idx = 0
        for tag, state in states.items():
            evolved_states = propagator * state
            # convert states back to sigma z basis:
            evolved_states = (evecs @ evolved_states.T).T
            entropy = l2main.get_entropy(evolved_states, L//2, note=f'{tag}_entropy_trace_L{L}')

            axes[ax_idx].plot(TSPACE, entropy, label=rf'$L={L}$')
            axes[ax_idx].set_xlabel(r'Time $t$')
            ax_idx += 1

    axes[0].set_title(r'$\otimes \frac{1}{2}(|\uparrow\rangle - \sqrt{3}|\downarrow\rangle)$')
    axes[1].set_title(r'$\otimes \frac{1}{\sqrt{2}}(-|\uparrow\rangle + |\downarrow\rangle)$')

    axes[0].set_ylabel(r'$S_{L/2}(t)$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'p4_1_3.png', **FIG_SAVE_OPTIONS)


if __name__ == '__main__':
    p4_1_12()

    # p4_1_3()


