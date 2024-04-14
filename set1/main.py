import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import utility
from collections import OrderedDict

plt.rcParams['font.size'] = 14

LSPACE = (8, 10, 12, 14)
HSPACE = np.linspace(0, 2, 17)
CACHE_DIR = './data/'
FIGS_DIR = './figs/'
LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
num_ext_states = 3


@utility.cache('npz', CACHE_DIR + 'dense_H')
def make_dense_H(L, hspace=HSPACE, note=None):
    dim = 2 ** L
    H_open = np.zeros((len(hspace), dim, dim))
    H_loop = np.zeros((len(hspace), dim, dim))
    for i in range(dim):
        H_open[:, i, i] = 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)
        for flip in range(0, L):
            H_open[:, i ^ (1 << flip), i] -= hspace
        H_loop[:, :, i] = H_open[:, :, i]
        H_loop[:, i, i] += 2 * ((i ^ (i >> (L - 1))) % 2) - 1
    return {'open': H_open, 'loop': H_loop}


@utility.cache('pkl', CACHE_DIR + 'sparse_H')
def make_sparse_H(L, hspace=HSPACE, note=None):
    # 0 for spin down, 1 for spin up.
    dim = 2 ** L
    H_open = []
    H_loop = []
    for i, h in enumerate(hspace):
        print(f'making sparse H: L={L}, h={h}')
        row = []
        col = []
        data_open = []
        data_loop = []
        for j in range(dim):
            row.append(j)
            col.append(j)
            data_open.append(2 * ((j & ~(1 << L-1)) ^ (j >> 1)).bit_count() - (L-1))
            data_loop.append(data_open[-1] + 2 * ((j ^ (j >> (L - 1))) % 2) - 1)
            for flip in range(0, L):
                row.append(j ^ (1 << flip))
                col.append(j)
                data_open.append(-h)
                data_loop.append(-h)
        H_open.append(sp.sparse.csr_matrix((data_open, (row, col)), shape=(dim, dim)))
        H_loop.append(sp.sparse.csr_matrix((data_loop, (row, col)), shape=(dim, dim)))
    return {'open': H_open, 'loop': H_loop}


@utility.cache('npz', CACHE_DIR + 'dense_eigs')
def dense_eigs(H, hspace=HSPACE, note=None):
    all_evals = []
    all_evecs = []
    for i in range(len(hspace)):
        print(f'finding dense evals: L={np.log2(H[i].shape[0])}, h={hspace[i]}')
        evals, evecs = sp.linalg.eigh(H[i])
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}


@utility.cache('npz', CACHE_DIR + 'sparse_eigs')
def sparse_eigs(H, hspace=HSPACE, note=None):
    all_evals = []
    all_evecs = []
    for i in range(len(hspace)):
        print(f'finding sparse evals: L={np.log2(H[i].shape[0])}, h={hspace[i]}')
        evals, evecs = sp.sparse.linalg.eigsh(H[i], k=1+num_ext_states, which='SA')
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}


@utility.cache('pkl', CACHE_DIR + 'sparse_H_sx')
def make_sparse_H_sx(L, hspace=HSPACE, note=None):
    # 0 for antisymmetric state, 1 for symmetric state.
    all_states = [OrderedDict(), OrderedDict()]
    for state in range(2**L):
        sign = state.bit_count() % 2
        idx = len(all_states[sign])
        all_states[sign][state] = idx
    H = []
    for i, h in enumerate(hspace):
        print(f'making sparse H in sx basis: L={L}, h={h}')
        H.append([])
        for sign, states in enumerate(all_states):
            row = []
            col = []
            data = []
            for j, state in enumerate(states):
                row.append(j)
                col.append(j)
                data.append(-h * (2 * state.bit_count() - L))
                for flip in range(L-1):
                    row.append(states[state ^ (0b11 << flip)])
                    col.append(j)
                    data.append(-1)
            H[i].append(sp.sparse.csr_matrix((data, (row, col)), shape=(len(states), len(states))))
    return H


@utility.cache('npz', CACHE_DIR + 'sparse_eigs_sx')
def sparse_eigs_sx(H, note=None):
    all_evals = []
    all_evecs = []
    for i, blocks in enumerate(H):
        print(f'finding sparse evals in sx basis: L={np.log2(blocks[0].shape[0]) + 1}, h={HSPACE[i]}')
        all_evals.append([])
        all_evecs.append([])
        for sign, block in enumerate(blocks):
            evals, evecs = sp.sparse.linalg.eigsh(block, k=1+num_ext_states, which='SA')
            all_evals[i].append(evals)
            all_evecs[i].append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}


def p4_1(Lspace=LSPACE):
    plt.figure()
    for C, L in enumerate(Lspace):
        gnd_eng = {}
        if L <= 12:
            H = make_dense_H(L, note=f'L{L}')
            for bdry in ('open', 'loop'):
                gnd_eng[bdry] = np.min(dense_eigs(H[bdry], note=f'{bdry}_L{L}')['evals'], axis=-1)
        else:
            for i, h in enumerate(HSPACE):
                fake_hspace = np.array([h])
                H = make_dense_H(L, hspace=fake_hspace, note=f'L{L}_h{round(h, 3)}')
                for bdry in ('open', 'loop'):
                    gnd_eng[bdry] = np.min(dense_eigs(H[bdry], hspace=fake_hspace, note=f'{bdry}_L{L}_h{round(h, 3)}')['evals'], axis=-1)[0]

        plt.plot(HSPACE, gnd_eng['open'], label=rf'$L={L}$', color=f'C{C}')
        plt.plot(HSPACE, gnd_eng['loop'], color=f'C{C}', linestyle='dotted')
    plt.xlabel(r'$h/J$')
    plt.ylabel(r'$E_0/J$')
    plt.legend()
    plt.savefig(FIGS_DIR + 'p4_1.png', bbox_inches='tight')


def p4_2(Lspace=LSPACE):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig_comp = {}
    ax_comp = {}
    for bdry in ('open', 'loop'):
        fig_comp[bdry], ax_comp[bdry] = plt.subplots()
    color = 0
    for L in Lspace:
        H = make_sparse_H(L, note=f'L{L}')
        evals = {}
        evecs = {}
        for bdry in H:
            data = sparse_eigs(H[bdry], note=f'{bdry}_L{L}')
            evals[bdry] = data['evals']
            evecs[bdry] = data['evecs']

        if L in np.arange(8, 21, 2):
            for i, ax in enumerate(axes.flatten()):
                ax.plot(HSPACE, evals['open'][:, i], label=rf'L={L}', color=f'C{color}')
                ax.plot(HSPACE, evals['loop'][:, i], color=f'C{color}', linestyle='dotted')
            color += 1
        for bdry in fig_comp:
            if L in p4_1_Lspace:
                dense_evals = np.min(np.load(CACHE_DIR + f'dense_eigs_{bdry}_L{L}.npz')['evals'], axis=-1)
                ax_comp[bdry].plot(HSPACE, np.abs(evals[bdry][:, 0] - dense_evals), label=rf'L={L}')
                ax_comp[bdry].set_yscale('log')
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(rf'$|{i}\rangle$')
        ax.set_xlabel(r'$h/J$')
        ax.set_ylabel(rf'$E_{i}/J$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + f'p4_2_evals.png', bbox_inches='tight')
    for bdry in fig_comp:
        ax_comp[bdry].set_xlabel(r'$h/J$')
        ax_comp[bdry].set_ylabel(r'Difference ($J$)')
        fig_comp[bdry].legend(**LEGEND_OPTIONS)
        fig_comp[bdry].savefig(FIGS_DIR + f'p4_2_{bdry}_comp.png', bbox_inches='tight')


def p4_3(Lspace=LSPACE):
    h = {'ferro': np.where(HSPACE == 0.25)[0][0], 'para': np.where(HSPACE == 0.75)[0][0]}
    gnd_eng = {}
    for mag in h:
        gnd_eng[mag] = {'open': [], 'loop': []}
    for L in Lspace:
        for mag in h:
            for bdry in gnd_eng[mag]:
                evals = np.load(CACHE_DIR + f'sparse_eigs_{bdry}_L{L}.npz')['evals']
                gnd_eng[mag][bdry].append(np.min(evals, axis=-1)[h[mag]] / L)

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    for i, mag in enumerate(h):
        for bdry in gnd_eng[mag]:
            if bdry == 'open':
                bulk_open = [None] * 2
                for j in range(2, len(gnd_eng[mag][bdry])):
                    bulk_open += [(gnd_eng[mag][bdry][j] * Lspace[j] - gnd_eng[mag][bdry][j-2] * Lspace[j-2])/2]
                axes[i].plot(Lspace, gnd_eng[mag][bdry], label='Open Boundary')
                axes[i].plot(Lspace, bulk_open, label='Bulk of Open Boundary:\n' + r'$(E_0(L) - E_0(L-2))/2J$')
            else:
                axes[i].plot(Lspace, gnd_eng[mag][bdry], label='Periodic Boundary')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, **LEGEND_OPTIONS)
        axes[i].set_title(rf'$h={HSPACE[h[mag]]}$')
        axes[i].set_xlabel(r'$L$')
        axes[i].set_ylabel(r'$E_0(L)/J$')
    plt.savefig(FIGS_DIR + 'p4_3_L_dep_of_gnd_eng.png', bbox_inches='tight')


def p4_4(Lspace = LSPACE):
    L = Lspace[-1]
    data = np.load(CACHE_DIR + f'sparse_eigs_loop_L{L}.npz')
    evals = data['evals']
    evecs = data['evecs']

    plt.figure()
    for i in range(1 + num_ext_states):
        plt.plot(HSPACE, evals[:, i], label=rf'$E_{i}$')
    plt.xlabel(r'$h/J$')
    plt.ylabel(r'$E_i/J$')
    plt.legend()
    plt.savefig(FIGS_DIR + f'p4_4_L{L}_spectrum_linear.png', bbox_inches='tight')

    plt.figure()
    plt.plot(HSPACE, np.abs(evals[:, 0] - evals[:, 1]))
    plt.xlabel(r'$h/J$')
    plt.ylabel(r'$|E_1 - E_0|/J$')
    plt.savefig(FIGS_DIR + f'p4_4_L{L}_gap.png', bbox_inches='tight')

    gnd_states = evecs[:, :, 0]
    fid = np.abs(np.sum(np.conj(gnd_states) * np.roll(gnd_states, -1, axis=0), axis=-1))[:-1]

    plt.figure()
    plt.plot(HSPACE[:-1], fid)
    plt.xlabel(r'$h/J$')
    plt.ylabel(rf'Fidelity, $\delta h = {HSPACE[1] - HSPACE[0]}$')
    plt.savefig(FIGS_DIR + f'p4_4_L{L}_fid.png', bbox_inches='tight')


def p4_5(Lspace=LSPACE):
    h = {
        'Ferromagnet': np.where(HSPACE == 0.25)[0][0],
        'Critical Point': np.where(HSPACE == 1)[0][0],
        'Paramagnet': np.where(HSPACE == 1.75)[0][0]
    }
    fig_order, axes_order = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    fig_correl, axes_correl = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    for C, L in enumerate(Lspace):
        gnd_states = np.load(CACHE_DIR + f'sparse_eigs_loop_L{L}.npz')['evecs'][:, :, 0]
        states = np.arange(2**L)
        exp_vals = np.matmul(np.abs(gnd_states)**2, (-2 * ((states % 2) ^ (states >> np.arange(L)[:, np.newaxis]) % 2) + 1).T)
        correl = exp_vals[:, :L//2]
        half_loop = correl[:, -1]
        M2 = np.sum(exp_vals, axis=-1) / L

        axes_order[0].plot(HSPACE, half_loop, label=rf'$L={L}$', color=f'C{C}')
        axes_order[1].plot(HSPACE, M2, label=rf'$L={L}$', color=f'C{C}')

        for mag_idx, mag in enumerate(h):
            axes_correl[mag_idx].plot(np.arange(correl.shape[-1]), correl[h[mag]], label=rf'$L={L}$', color=f'C{C}')
            axes_correl[mag_idx].set_title(mag + rf': $h/J={HSPACE[h[mag]]}$')
            axes_correl[mag_idx].set_xlabel(r'$r$')
            axes_correl[mag_idx].set_ylabel(r'$C^{zz}(r)$')

    axes_order[0].set_ylabel(r'$\langle \sigma_1^z \sigma_{L/2}^z \rangle$')
    axes_order[1].set_ylabel(r'$\langle (M/L)^2 \rangle$')
    for ax in axes_order:
        ax.set_xlabel(r'$h/J$')
    handles, labels = axes_order[0].get_legend_handles_labels()
    fig_order.legend(handles, labels, **LEGEND_OPTIONS)
    fig_order.savefig(FIGS_DIR + 'p4_5_order.png', bbox_inches='tight')

    handles, labels = axes_correl[0].get_legend_handles_labels()
    fig_correl.legend(handles, labels, **LEGEND_OPTIONS)
    fig_correl.savefig(FIGS_DIR + 'p4_5_correl.png', bbox_inches='tight')


def p4_7(Lspace=LSPACE):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    h = {
        'Ferromagnet': np.where(HSPACE == 0.25)[0][0],
        'Critical Point': np.where(HSPACE == 1)[0][0],
        'Paramagnet': np.where(HSPACE == 1.75)[0][0]
    }
    splitting = {}
    for mag in h:
        splitting[mag] = []
    color = 0
    for L in Lspace:
        print(L)
        H = make_sparse_H_sx(L, hspace=HSPACE, note=f'L{L}')
        eigs = sparse_eigs_sx(H, note=f'L{L}')
        evals = eigs['evals']
        minus_evals = evals[:, 0, :]
        plus_evals = evals[:, 1, :]
        for mag in h:
            splitting[mag].append(np.abs(plus_evals[h[mag], 0] - minus_evals[h[mag], 0]))
        # combined_evals = np.concatenate((minus_evals, plus_evals), axis=-1)
        # # Use the last value of h to sort because the sorting should be independent of h
        # sort = np.argsort(combined_evals[-1], axis=-1)
        # if L in np.arange(8, 22, 2):
        #     for i, ax in enumerate(axes.flatten()):
        #         ax.plot(HSPACE, combined_evals[:, sort[i]], label=rf'$L={L}$', color=f'C{color}')
        #         if sort[i] < (1+num_ext_states):
        #             ax.set_title(rf'$|{i}\rangle$, from $(-)$ Sector')
        #         else:
        #             ax.set_title(rf'$|{i}\rangle$, from $(+)$ Sector')
        #         ax.set_xlabel(r'$h/J$')
        #         ax.set_ylabel(rf'$E_{i}/J$')
        #     color += 1
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + f'p4_7_evals_sx.png', bbox_inches='tight')

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    for i, mag in enumerate(h):
        axes[i].plot(Lspace, splitting[mag])
        axes[i].set_title(rf'${mag}: h/J={h}$')
        axes[i].set_xlabel(r'$L$')
        axes[i].set_ylabel(f'(E_1 - E_0)/J')
    fig.savefig(FIGS_DIR + f'p4_7_splitting_sx.png', bbox_inches='tight')


if __name__ == '__main__':
    p4_1_Lspace = np.array([8, 10, 12])
    p4_2_Lspace = np.arange(5, 21)
    p4_5_Lspace = np.arange(6, 22, 2)
    p4_7_Lspace = np.arange(5, 22)

    # p4_1(Lspace=p4_1_Lspace)

    # p4_2(Lspace=p4_2_Lspace)

    # p4_3(Lspace=p4_2_Lspace)

    # p4_4(Lspace=p4_2_Lspace)

    # p4_5(Lspace=p4_5_Lspace)

    p4_7(Lspace=p4_7_Lspace)

    # plt.show()
