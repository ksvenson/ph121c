import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from lab2 import main as l2main
import utility

CACHE_DIR = './data/'
FIGS_DIR = './figs/'

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}

LSPACE = np.arange(8, 14, 2)
TSPACE = np.linspace(0, 50, 100)
beta_space = np.linspace(-3, 3, 100)

FIELD_VALS = {'hx': -1.05, 'hz': 0.5}

# Pauli matrices defined in a basis where first component is spin down, second component is spin up.
ops = {
    'x': np.array([[0, 1], [1, 0]]),
    'y': np.array([[0, 1j], [-1j, 0]]),
    'z': np.array([[-1, 0], [0, 1]])
}


@utility.cache('npy', CACHE_DIR + 'l3_dense_H')
def make_dense_H(L, W=None, note=None):
    """
    Makes Hamiltonian in sigma_z basis as defined in Equation 8 and 11 with periodic boundary conditions.
    States are indexed in binary: 0 for spin down and 1 for spin up.
    :param L: System size.
    :param W: Uniform distribution parameter.
    :param note: Appeneded to filename when caching function output.
    :return: Hamiltonian with shape (2**L, 2**L).
    """
    H = np.zeros((2**L, 2**L))
    for i in range(2**L):
        # Perform a XOR between states, and states shifted by 1 bit.
        H[i, i] += 2 * (i ^ cycle_bits(i, L, 1)).bit_count() - L

        flips = np.arange(L)
        if W is None:
            # h_z field
            H[i, i] += -1 * FIELD_VALS['hz'] * (2 * i.bit_count() - L)

            # h_x field
            H[i ^ (1 << flips), i] += -1 * FIELD_VALS['hx']
        else:
            hx = np.random.uniform(low=-W, high=W, size=L)
            hz = np.random.uniform(low=-W, high=W, size=L)

            # h_z field
            sigma_z = 2 * np.array([int(bit) for bit in format(i, f'0{L}b')]) - 1
            H[i, i] += -1 * np.sum(hz * sigma_z)

            # h_x field
            H[i ^ (1 << flips), i] += -1 * hx
    return H


def make_scar_H(L, Omega, note=None):
    H = np.zeros((2**L, 2**L))
    for i in range(2**L):
        flips = np.arange(L)

        # Omega term
        H[i ^ (1 << flips), i] += Omega/2

        # 1/4 term in P_{j, j+1}
        H[i, i] += (1/4) * (2 * i.bit_count() - L)

        # sigma_x and sigma_y terms in P_{j, j+1}
        for j in range(1, L-1):
            new_state = i ^ (0b11 << j)
            j_bit = 2 * ((i >> (j+1)) & 1) - 1
            j1_bit = 2 * ((i >> j) & 1) - 1
            j2_bit = 2 * ((i >> (j-1)) & 1) - 1
            H[new_state, i] += (-1/4) * j2_bit * (1 + (-1) * j_bit * j1_bit)

        j_bit = 2 * ((i >> 1) & 1) - 1
        j1_bit = 2 * (i & 1) - 1
        j2_bit = 2 * ((i >> (L-1)) & 1) - 1
        H[i ^ 0b11, i] += (-1/4) * j2_bit * (1 + (-1) * j_bit * j1_bit)

        j_bit = 2 * (i & 1) - 1
        j1_bit = 2 * ((i >> (L-1)) & 1) - 1
        j2_bit = 2 * ((i >> (L-2)) & 1) - 1
        H[i ^ 0b1 ^ (1 << (L-1)), i] += (-1/4) * j2_bit * (1 + (-1) * j_bit * j1_bit)

        # sigma_z term in P_{j, j+1}
        H[i, i] += (1/4) * (2 * (i ^ cycle_bits(i, L, 1) ^ cycle_bits(i, L, 2)).bit_count() - L)


@utility.cache('npz', CACHE_DIR + 'l3_dense_eigs')
def dense_eigs(L, W=None, note=None):
    """
    Diagonalizes Hamiltonian from `make_dense_H`.
    :param L: System size.
    :param note: Appended to filename when caching function output.
    :return: As a dict: list of evals, matrix with evecs as columns.
    """
    print(f'finding dense evals: L={L}')
    H = make_dense_H(L, W=W, note=note)
    evals, evecs = sp.linalg.eigh(H)
    return {'evals': evals, 'evecs': evecs}


def cycle_bits(bits, L, l):
    """
    Cycles the bits in `bits` by `l` positions to the right.
    For example, `cycle_bits(0b101, 3, 1)` returns `0b110`.
    :param bits: Bit string.
    :param L: Length of bit string (to know how many leading zeros there are)
    :return: Cycled bit string.
    """
    return ((bits >> l) | (bits << (L-l))) & (2**L - 1)


def make_prod_state(L, up_coeff, down_coeff):
    """
    Make a translation-invariant product state in the sigma z basis.
    :param L: System size.
    :param up_coeff: Spin up coefficient.
    :param down_coeff: Spin down coefficient.
    :return: length 2**L array representing the state.
    """
    output = []
    for state in range(2**L):
        num_up = state.bit_count()
        num_down = L - num_up
        output.append((up_coeff**num_up) * (down_coeff**num_down))
    return output


def globalize_op(L, op):
    """
    Take local 2x2 operator `op` and turn it into a global 2**L x 2**L operator in sigma z basis.
    :param L: System size.
    :param op: 2x2 local operator at site 1.
    :return: 2**L x 2**L operator in sigma z basis.
    """
    return np.kron(op, np.identity(2**(L-1)))


def rebase_operator(L, op, evecs):
    """
    First, take local 2x2 operator `op` and turn it into a global 2**L x 2**L operator in sigma z basis.
    Then, change basis of global operator to energy basis using `evecs`.
    :param L: System size.
    :param op: 2x2 local operator at site 1.
    :return: 2**L x 2**L operator in energy basis.
    """
    return evecs.T.conj() @ globalize_op(L, op) @ evecs


def make_trans_op(L):
    """
    Creates the translation operator as defined in 4.2.1. in the sigma z basis.
    :param L: System size.
    :return: Translation operator.
    """
    T = np.zeros((2**L, 2**L))
    states = np.arange(2**L)
    T[cycle_bits(states, L, 1), states] = 1
    return T


# def ham_analysis(prob, W=None):
#     """
#     Makes all necessary plots for 4.1.1., 4.1.2, and 4.3.1. Leave `W=None` for translation symmetry.
#     :param prob: A tag used to name figures.
#     :param W: Uniform distribution parameter
#     :return: Nothing. Saves plots to `FIGS_DIR`.
#     """
#     fig_sig, axes_sig = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(15, 15))
#     fig_beta, axes_beta = plt.subplots(figsize=(5, 5))
#     fig_entropy, axes_entropy = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
#     for color_idx, L in enumerate(LSPACE):
#         print(f'L={L}')
#         eigs = dense_eigs(L, W=W, note=f'{prob}_L{L}')
#         evals = eigs['evals']
#         evecs = eigs['evecs']
#
#         xi_state = make_prod_state(L, 1/2, -np.sqrt(3)/2)
#         # convert xi state to diagonal basis
#         xi_state = evecs.T.conj() @ xi_state
#
#         propagator = np.exp(-1j * np.multiply.outer(TSPACE, evals))
#         # This method directly evolves the state like in Equation 2, which I found to be faster than
#         # evolving the expectation value, like in Equation 3.
#         evolved_states = (propagator * xi_state).T
#
#         # Thermal calculations
#         Z_beta = np.array([np.sum(np.exp(-beta * evals)) for beta in beta_space])
#         E_beta = np.array([np.sum(np.exp(-beta * evals) * evals) for beta in beta_space]) / Z_beta
#         xi_eng = np.sum(xi_state.conj() * evals * xi_state)
#         xi_beta_idx = np.argmin(np.abs(E_beta - xi_eng))
#         xi_beta = beta_space[xi_beta_idx]
#
#         for op_idx, op in enumerate(ops):
#             print(f'op: {op}')
#             global_op = rebase_operator(L, ops[op], evecs)
#             measurement = np.sum(evolved_states.conj() * (global_op @ evolved_states), axis=0)
#
#             O_thermal = np.sum(np.diag(global_op) * np.exp(-1 * xi_beta * evals)) / Z_beta[xi_beta_idx]
#
#             axes_sig[op_idx].plot(TSPACE, measurement, label=f'$L={L}$', color=f'C{color_idx}')
#             axes_sig[op_idx].axhline(O_thermal, label=rf'$L={L}$ Thermal Limit', color=f'C{color_idx}', linestyle='dotted')
#             axes_sig[op_idx].set_ylabel(rf'$\langle \sigma_1^{op}(t) \rangle$')
#             axes_sig[op_idx].set_title(rf'$\mu={op}$')
#         axes_beta.plot(beta_space, E_beta, label=rf'$L={L}$')
#
#         # Entropy calculations
#         my_state = make_prod_state(L, -1/np.sqrt(2), 1/np.sqrt(2))
#         my_state = evecs.T.conj() @ my_state
#         my_evolved_states = (propagator * my_state).T
#         all_states = {'xi': evolved_states, 'my': my_evolved_states}
#         ax_idx = 0
#         for tag, states in all_states.items():
#             # convert states back to sigma z basis:
#             sigz_states = evecs @ states
#             entropy = l2main.get_entropy(sigz_states.T, L//2, note=f'{prob}_L{L}_{tag}_entropy_trace')
#
#             axes_entropy[ax_idx].plot(TSPACE, entropy, label=rf'$L={L}$')
#             axes_entropy[ax_idx].set_xlabel(r'Time $t$')
#             ax_idx += 1
#
#     axes_sig[-1].set_xlabel('Time $t$')
#     handles, labels = axes_sig[0].get_legend_handles_labels()
#     fig_sig.legend(handles, labels, **LEGEND_OPTIONS)
#     fig_sig.savefig(FIGS_DIR + f'{prob}_sigma.png', **FIG_SAVE_OPTIONS)
#
#     axes_beta.set_xlabel(r'$\beta$')
#     axes_beta.set_ylabel(r'$E_\beta$')
#     fig_beta.legend(**LEGEND_OPTIONS)
#     fig_beta.savefig(FIGS_DIR + f'{prob}_E_beta.png', **FIG_SAVE_OPTIONS)
#
#     axes_entropy[0].set_title(r'$\otimes \frac{1}{2}(|\uparrow\rangle - \sqrt{3}|\downarrow\rangle)$')
#     axes_entropy[1].set_title(r'$\otimes \frac{1}{\sqrt{2}}(-|\uparrow\rangle + |\downarrow\rangle)$')
#     axes_entropy[0].set_ylabel(r'$S_{L/2}(t)$')
#     handles, labels = axes_entropy[0].get_legend_handles_labels()
#     fig_entropy.legend(handles, labels, **LEGEND_OPTIONS)
#     fig_entropy.savefig(FIGS_DIR + f'{prob}_entropy.png', **FIG_SAVE_OPTIONS)


def do_ham_analysis(prob, W=None):
    """
    Computes all necessary data for 4.1.1., 4.1.2, and 4.3.1. Leave `W=None` for translation symmetry.
    :param prob: A tag used to name figures.
    :param W: Uniform distribution parameter
    :return: A dictionary of data.
    """
    data = {}
    for color_idx, L in enumerate(LSPACE):
        print(f'L={L}')
        data[L] = {
            'sigma': {},
            'entropy': {}
        }
        eigs = dense_eigs(L, W=W, note=f'{prob}_L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']

        xi_state = make_prod_state(L, 1/2, -np.sqrt(3)/2)
        # convert xi state to diagonal basis
        xi_state = evecs.T.conj() @ xi_state

        propagator = np.exp(-1j * np.multiply.outer(TSPACE, evals))
        # This method directly evolves the state like in Equation 2, which I found to be faster than
        # evolving the expectation value, like in Equation 3.
        evolved_states = (propagator * xi_state).T

        # Thermal calculations
        Z_beta = np.array([np.sum(np.exp(-beta * evals)) for beta in beta_space])
        E_beta = np.array([np.sum(np.exp(-beta * evals) * evals) for beta in beta_space]) / Z_beta
        xi_eng = np.sum(xi_state.conj() * evals * xi_state)
        xi_beta_idx = np.argmin(np.abs(E_beta - xi_eng))
        xi_beta = beta_space[xi_beta_idx]
        data[L]['E_beta'] = E_beta

        for op_idx, op in enumerate(ops):
            print(f'op: {op}')
            global_op = rebase_operator(L, ops[op], evecs)
            measurement = np.sum(evolved_states.conj() * (global_op @ evolved_states), axis=0)
            data[L]['sigma'][op] = measurement

            O_thermal = np.sum(np.diag(global_op) * np.exp(-1 * xi_beta * evals)) / Z_beta[xi_beta_idx]
            data[L]['sigma'][f'{op}_thermal'] = O_thermal

        # Entropy calculations
        my_state = make_prod_state(L, -1/np.sqrt(2), 1/np.sqrt(2))
        my_state = evecs.T.conj() @ my_state
        my_evolved_states = (propagator * my_state).T
        all_states = {'xi': evolved_states, 'my': my_evolved_states}
        for tag, states in all_states.items():
            # convert states back to sigma z basis:
            sigz_states = evecs @ states
            entropy = l2main.get_entropy(sigz_states.T, L//2, note=f'{prob}_L{L}_{tag}_entropy_trace')

            data[L]['entropy'][tag] = entropy

    return data


def plot_ham_analysis(prob, data):
    """
    Makes all necessary plots for 4.1.1., 4.1.2, and 4.3.1. using the data generated in `do_ham_analysis`.
    :param prob: A tag used to name figures.
    :param data: Output of `do_ham_analysis'.
    :return: Nothing. Saves plots to `FIGS_DIR`.
    """
    fig_sig, axes_sig = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(15, 15))
    fig_beta, axes_beta = plt.subplots(figsize=(5, 5))
    fig_entropy, axes_entropy = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    for color_idx, L in enumerate(LSPACE):
        for op_idx, op in enumerate(ops):
            axes_sig[op_idx].plot(TSPACE, data[L]['sigma'][op], label=f'$L={L}$', color=f'C{color_idx}')
            axes_sig[op_idx].axhline(data[L]['sigma'][f'{op}_thermal'], label=rf'$L={L}$ Thermal Limit', color=f'C{color_idx}', linestyle='dotted')
            axes_sig[op_idx].set_ylabel(rf'$\langle \sigma_1^{op}(t) \rangle$')
            axes_sig[op_idx].set_title(rf'$\mu={op}$')

        axes_beta.plot(beta_space, data[L]['E_beta'], label=rf'$L={L}$')

        for idx, tag in enumerate(data[L]['entropy']):
            axes_entropy[idx].plot(TSPACE, data[L]['entropy'][tag], label=rf'$L={L}$')
            axes_entropy[idx].set_xlabel(r'Time $t$')

    axes_sig[-1].set_xlabel('Time $t$')
    handles, labels = axes_sig[0].get_legend_handles_labels()
    fig_sig.legend(handles, labels, **LEGEND_OPTIONS)
    fig_sig.savefig(FIGS_DIR + f'{prob}_sigma.png', **FIG_SAVE_OPTIONS)

    axes_beta.set_xlabel(r'$\beta$')
    axes_beta.set_ylabel(r'$E_\beta$')
    fig_beta.legend(**LEGEND_OPTIONS)
    fig_beta.savefig(FIGS_DIR + f'{prob}_E_beta.png', **FIG_SAVE_OPTIONS)

    axes_entropy[0].set_title(r'$\otimes \frac{1}{2}(|\uparrow\rangle - \sqrt{3}|\downarrow\rangle)$')
    axes_entropy[1].set_title(r'$\otimes \frac{1}{\sqrt{2}}(-|\uparrow\rangle + |\downarrow\rangle)$')
    axes_entropy[0].set_ylabel(r'$S_{L/2}(t)$')
    handles, labels = axes_entropy[0].get_legend_handles_labels()
    fig_entropy.legend(handles, labels, **LEGEND_OPTIONS)
    fig_entropy.savefig(FIGS_DIR + f'{prob}_entropy.png', **FIG_SAVE_OPTIONS)


def p4_1_123():
    prob = 'p4_1_123'
    plot_ham_analysis(prob, do_ham_analysis(prob, W=None))


def p4_2_12():
    fig_expval, axes_expval = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    fig_entropy, axes_entropy = plt.subplots(figsize=(10, 5))
    for L in LSPACE:
        print(f'L={L}')
        eigs = dense_eigs(L, note=f'L{L}')
        evals = eigs['evals']
        evecs = eigs['evecs']

        T = make_trans_op(L)
        T_expvals = np.sum(evecs.conj() * (T @ evecs), axis=0)
        k0_sector = np.abs(T_expvals - 1) <= 1e-5

        k0_evals = evals[k0_sector]
        k0_evecs = evecs[:, k0_sector]

        for op_idx, op in enumerate(ops):
            print(f'op: {op}')
            global_op = globalize_op(L, ops[op])
            sigma_expvals = np.sum(k0_evecs.conj() * (global_op @ k0_evecs), axis=0)

            axes_expval[op_idx].plot(k0_evals/L, sigma_expvals, label=rf'$L={L}$')
            axes_expval[op_idx].set_xlabel(r'$\varepsilon_n / L$')
            axes_expval[op_idx].set_title(rf'$\mu={op}$')

        entropy = l2main.get_entropy(k0_evecs.T, L//2, note=f'p4_2_2_entropy_L{L}')

        axes_entropy.plot(k0_evals / L, entropy/L, label=rf'$L={L}$')
        axes_entropy.set_xlabel(r'$\varepsilon_n / L$')

    axes_expval[0].set_ylabel(r'$\langle \sigma_1^\mu \rangle_n$')
    handles, labels = axes_expval[0].get_legend_handles_labels()
    fig_expval.legend(handles, labels, **LEGEND_OPTIONS)
    fig_expval.savefig(FIGS_DIR + 'p4_2_1.png', **FIG_SAVE_OPTIONS)

    axes_entropy.set_ylabel(r'$S_{L/2} / L$')
    handles, labels = axes_entropy.get_legend_handles_labels()
    fig_entropy.legend(handles, labels, **LEGEND_OPTIONS)
    fig_entropy.savefig(FIGS_DIR + 'p4_2_2.png', **FIG_SAVE_OPTIONS)


def p4_3_1():
    for W in (10,):
        data = []
        trials = np.arange(10)
        for trial in trials:
            print(f'Trial {trial}')
            prob = f'p4_2_1_W{W}_trial{trial}'
            data.append(do_ham_analysis(prob, W=W))
            plot_ham_analysis(prob, data[-1])
        avg = {}
        for L in data[0]:
            avg[L] = {
                'sigma': {},
                'entropy': {}
            }
            for op in ops:
                avg[L]['sigma'][op] = np.mean([data[i][L]['sigma'][op] for i in trials], axis=0)
                avg[L]['sigma'][f'{op}_thermal'] = np.mean([data[i][L]['sigma'][f'{op}_thermal'] for i in trials])

            avg[L]['E_beta'] = np.mean([data[i][L]['E_beta'] for i in trials], axis=0)

            for tag in data[0][L]['entropy']:
                avg[L]['entropy'][tag] = np.mean([data[i][L]['entropy'][tag] for i in trials], axis=0)

        plot_ham_analysis(f'p4_2_1_W{W}_avg', avg)


def testing():
    L = 8
    no_W = make_dense_H(L, W=None)
    with_W = make_dense_H(L, W=0)
    print(f'Error: {np.linalg.norm(no_W - with_W)}')


if __name__ == '__main__':
    np.random.seed(628)  # tau = 2 pi - the better circle constant!

    # p4_1_123()

    # p4_2_12()

    p4_3_1()

    # testing()

