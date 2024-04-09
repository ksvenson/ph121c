import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import utility

plt.rcParams['font.size'] = 14

LSPACE = (8, 10, 12, 14)
HSPACE = np.linspace(-2, 2, 17)
CACHE_DIR = './data/'
FIGS_DIR = './figs/'
LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}


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
        evals, evecs = sp.linalg.eig(H[i])
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}


@utility.cache('npz', CACHE_DIR + 'sparse_eigs')
def sparse_eigs(H, hspace=HSPACE, note=None):
    all_evals = []
    all_evecs = []
    for i in range(len(hspace)):
        print(f'finding sparse evals: L={np.log2(H[i].shape[0])}, h={hspace[i]}')
        evals, evecs = sp.sparse.linalg.eigs(H[i], k=4, which='SR')
        all_evals.append(evals)
        all_evecs.append(evecs)
    return {'evals': np.array(all_evals), 'evecs': np.array(all_evecs)}


# 4.1
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
    plt.legend(**LEGEND_OPTIONS)
    plt.savefig(FIGS_DIR + 'p4_1.png', bbox_inches='tight')


def p4_2(Lspace=LSPACE):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig_comp = {}
    ax_comp = {}
    for bdry in ('open', 'loop'):
        fig_comp[bdry], ax_comp[bdry] = plt.subplots()

    for C, L in enumerate(Lspace):
        H = make_sparse_H(L, note=f'L{L}')
        evals = {}
        evecs = {}
        for bdry in ('open', 'loop'):
            data = sparse_eigs(H[bdry], note=f'{bdry}_L{L}')
            evals[bdry] = data['evals']
            evecs[bdry] = data['evecs']

        if L in np.arange(8, 21, 2):
            for i, ax in enumerate(axes.flatten()):
                ax.plot(HSPACE, evals['open'][:, i], label=rf'L={L}', color=f'C{C}')
                ax.plot(HSPACE, evals['loop'][:, i], color=f'C{C}', linestyle='dotted')
        for bdry in ('open', 'loop'):
            if L in p4_1_Lspace:
                dense_H = make_dense_H(L, note=f'L{L}')
                dense_evals = np.min(dense_eigs(dense_H[bdry], note=f'{bdry}_L{L}')['evals'], axis=-1)
                ax_comp[bdry].plot(HSPACE, np.abs(evals[bdry][:, 0] - dense_evals), label=rf'L={L}')
                ax_comp[bdry].set_yscale('log')
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(rf'$|{i}\rangle$')
        ax.set_xlabel(r'$h/J$')
        ax.set_ylabel(rf'$E_{i}/J$')
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + f'p4_2_evals.png', bbox_inches='tight')
    for bdry in ('open', 'loop'):
        ax_comp[bdry].set_xlabel(r'$h/J$')
        ax_comp[bdry].set_ylabel(r'Difference ($J$)')
        fig_comp[bdry].legend(**LEGEND_OPTIONS)
        fig_comp[bdry].savefig(FIGS_DIR + f'p4_2_{bdry}_comp.png', bbox_inches='tight')


def p4_3(Lspace=LSPACE):
    f_h = 0.25
    p_h = 1.75


if __name__ == '__main__':
    with open('output.txt', 'w') as output:
        p4_1_Lspace = np.array([8, 10, 12])
        p4_2_Lspace = np.arange(5, 21)

        print('4.1.' + '-' * 50, file=output)
        p4_1(Lspace=p4_1_Lspace)

        print('4.2.' + '-' * 50, file=output)
        p4_2(Lspace=p4_2_Lspace)

    # plt.show()
