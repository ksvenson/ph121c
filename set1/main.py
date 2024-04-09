import matplotlib
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import utility

plt.rcParams['font.size'] = 14

LSPACE = (8, 10, 12, 14)
HSPACE = np.linspace(-2, 2, 10)
CACHE_DIR = './data/'


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
        H_loop[:, i, i] += 2 * ((i ^ (i >> L - 1)) % 2) - 1
    return {'H_open': H_open, 'H_loop': H_loop}


@utility.cache('pkl', CACHE_DIR + 'sparse_H')
def make_sparse_H(L, hspace=HSPACE, note=None):
    dim = 2 ** L
    output = []
    for i, h in enumerate(hspace):
        print(f'making sparse H: L={L}, h={h}')
        row = []
        col = []
        data = []
        for j in range(dim):
            row.append(j)
            col.append(j)
            data.append(2 * ((j & ~(1 << L-1)) ^ (j >> 1)).bit_count() - (L-1))
            for flip in range(0, L):
                row.append(j ^ (1 << flip))
                col.append(j)
                data.append(-h)
        output.append(sp.sparse.csr_matrix((data, (row, col)), shape=(dim, dim)))
    return output


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
        if L <= 12:
            data = make_dense_H(L, note=f'L{L}')
            H_open = data['H_open']
            H_loop = data['H_loop']
            gnd_eng_open = np.min(dense_eigs(H_open, note=f'open_L{L}')['evals'], axis=-1)
            gnd_eng_loop = np.min(dense_eigs(H_loop, note=f'loop_L{L}')['evals'], axis=-1)
        else:
            gnd_eng_open = []
            gnd_eng_loop = []
            for i, h in enumerate(HSPACE):
                fake_hspace = np.array([h])
                data = make_dense_H(L, hspace=fake_hspace, note=f'L{L}_h{round(h, 3)}')
                H_open = data['H_open']
                H_loop = data['H_loop']
                gnd_eng_open.append(np.min(dense_eigs(H_open, hspace=fake_hspace, note=f'open_L{L}_h{round(h, 3)}')['evals'], axis=-1)[0])
                gnd_eng_loop.append(np.min(dense_eigs(H_loop, hspace=fake_hspace, note=f'loop_L{L}_h{round(h, 3)}')['evals'], axis=-1)[0])

        plt.plot(HSPACE, gnd_eng_open, label=rf'$L={L}$', color=f'C{C}')
        plt.plot(HSPACE, gnd_eng_loop, color=f'C{C}', linestyle='dotted')
    # plt.title('Ground State Energy - Dense Method')
    plt.xlabel(r'$h/J$')
    plt.ylabel(r'$E_0/J$')
    plt.tight_layout()
    plt.legend()


def p4_2(Lspace=LSPACE):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    # figs = [plt.figure() for _ in range(4)]
    # axes = [fig.subplots() for fig in figs]

    for C, L in enumerate(Lspace):
        H = make_sparse_H(L, note=f'L{L}')
        data = sparse_eigs(H, note=f'L{L}')
        evals = data['evals']
        evecs = data['evecs']

        if L in np.arange(8, 21, 2):
            for i, ax in enumerate(axes.flatten()):
                ax.plot(HSPACE, evals[:, i], label=rf'L={L}')

    # fig.suptitle('Ground and Excited State Energies - Sparse Method')
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(rf'$|{i}\rangle$')
        ax.set_xlabel(r'$h/J$')
        ax.set_ylabel(rf'$E_{i}/J$')
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


if __name__ == '__main__':
    with open('output.txt', 'w') as output:
        p4_1_Lspace = np.array([8, 10, 12])
        p4_2_Lspace = np.arange(5, 21)

        print('4.1.' + '-' * 50, file=output)
        p4_1(Lspace=p4_1_Lspace)

        print('4.2.' + '-' * 50, file=output)
        p4_2(Lspace=p4_2_Lspace)

    plt.show()
