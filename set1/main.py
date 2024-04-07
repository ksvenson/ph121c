import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

LSPACE = (8, 10, 12, 14)
HSPACE = np.linspace(-2, 2, 10)
p4_1_fnames = ['./data/dense_gnd_eng_open', './data/dense_gnd_eng_loop']


def load_or_make(fname, make_func, multi=False, *args, **kwargs):
    if os.path.isfile(fname):
        file = np.load(fname)
        data = dict(file)
        file.close()
        return data
    else:
        data = make_func(*args, **kwargs)
        if multi:
            np.savez(fname, **data)
        else:
            np.save(fname, data)
        return data

def make_H(L, hspace=HSPACE):
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


def make_sparse_H(L, hspace=HSPACE):
    H_open, H_loop = make_H(L, hspace)
    return ([sp.sparse.csr_matrix(H_open[i]) for i in range(len(hspace))],
            [sp.sparse.csr_matrix(H_loop[i]) for i in range(len(hspace))])


def dense_evals(H, hspace=HSPACE):
    gnd_eng = []
    for i in range(len(hspace)):
        evals = sp.linalg.eigvals(H[i])
        gnd_eng.append(np.min(evals))
    return gnd_eng


def sparse_eigs(H, hspace=HSPACE):
    gnd_eng = []
    for i in range(len(hspace)):
        evals, evecs = sp.sparse.linalg.eigs(H[i], k=4, which='SR')
        gnd_eng.append(np.min(evals))
    return


# 4.1
def p4_1(Lspace=LSPACE):
    plt.figure()
    for C, L in enumerate(Lspace):
        H_open, H_loop = make_H(L)
        gnd_eng_open = load_or_make(f'./data/dense_gnd_eng_open_L{L}.npy', dense_evals, H_open)
        gnd_eng_loop = load_or_make(f'./data/dense_gnd_eng_loop_L{L}.npy', dense_evals, H_loop)

        plt.plot(HSPACE, gnd_eng_open, label=rf'$L={L}$', color=f'C{C}')
        plt.plot(HSPACE, gnd_eng_loop, color=f'C{C}', linestyle='dotted')
    plt.legend()
    plt.show()


def p4_2(Lspace=LSPACE):
    for C, L in enumerate(Lspace):
        H = make_sparse_H(L)[0]
        gnd_eng = load_or_make(f'./data/sparse_gnd_eng_open_L{L}.npy', )



if __name__ == '__main__':
    with open ('output.txt', 'w') as output:
        print('4.1.' + '-' * 50, file=output)
        p4_1(Lspace=(8, 10))
        # p4_1(Lspace=(8, 10, 12, 14, 20))
