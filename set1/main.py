import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import utility

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
        data = make_dense_H(L, note=f'L{L}')
        H_open = data['H_open']
        H_loop = data['H_loop']
        gnd_eng_open = np.min(dense_eigs(H_open, note=f'open_L{L}')['evals'], axis=-1)
        gnd_eng_loop = np.min(dense_eigs(H_loop, note=f'loop_L{L}')['evals'], axis=-1)

        plt.plot(HSPACE, gnd_eng_open, label=rf'$L={L}$', color=f'C{C}')
        plt.plot(HSPACE, gnd_eng_loop, color=f'C{C}', linestyle='dotted')
    plt.legend()


def p4_2(Lspace=LSPACE):
    plt.figure()
    for C, L in enumerate(Lspace):
        H = make_sparse_H(L, note=f'L{L}')
        data = sparse_eigs(H, note=f'L{L}')
        evals = data['evals']
        evecs = data['evecs']
        gnd_eng = np.min(evals, axis=-1)

        plt.plot(HSPACE, gnd_eng, label=rf'L={L}')
    plt.legend()


if __name__ == '__main__':
    with open('output.txt', 'w') as output:
        print('4.1.' + '-' * 50, file=output)
        p4_1(Lspace=(8, 10, 12, 14))

        print('4.2.' + '-' * 50, file=output)
        p4_2(Lspace=(8, 10, 12, 14, 16))
    plt.show()
