import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import pickle

LSPACE = (8, 10, 12, 14)
HSPACE = np.linspace(-2, 2, 10)
p4_1_fnames = ['./data/dense_gnd_eng_open', './data/dense_gnd_eng_loop']


def load_or_make(fname, make_func, *args, **kwargs):
    multi = fname.endswith('.npz')
    if os.path.isfile(fname):
        file = np.load(fname)
        if multi:
            data = dict(file)
            file.close()
        else:
            data = file
        return data
    else:
        data = make_func(*args, **kwargs)
        if multi:
            np.savez(fname, **data)
        else:
            np.save(fname, data)
        return data


def load_or_make_pkl(fname, make_func, *args, **kwargs):
    if os.path.isfile(fname):
        with open(fname, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        data = make_func(*args, **kwargs)
        with open(fname, 'wb') as file:
            pickle.dump(data, file)
        return data


def pkl_cache()


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


def dense_evals(H, hspace=HSPACE):
    gnd_eng = []
    for i in range(len(hspace)):
        print(f'finding dense evals: L={np.log2(H[i].shape[0])}, h={hspace[i]}')
        evals = sp.linalg.eigvals(H[i])
        gnd_eng.append(np.min(evals))
    return gnd_eng


def sparse_eigs(H, hspace=HSPACE):
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
            data = load_or_make(f'./data/dense_H_L{L}.npz', make_H, L)
            H_open = data['H_open']
            H_loop = data['H_loop']
            gnd_eng_open = load_or_make(f'./data/dense_gnd_eng_open_L{L}.npy', dense_evals, H_open)
            gnd_eng_loop = load_or_make(f'./data/dense_gnd_eng_loop_L{L}.npy', dense_evals, H_loop)
        else:
            gnd_eng_loop = []
            gnd_eng_open = []
            for i, h in enumerate(HSPACE):
                temp_hspace = np.array([h])
                data = make_H(L, hspace=temp_hspace)
                H_open = data['H_open']
                H_loop = data['H_loop']
                gnd_eng_open.append(load_or_make(f'./data/dense_gnd_eng_open_L{L}_h{i}.npy',
                                                 dense_evals, H_open, temp_hspace)[0])
                gnd_eng_loop.append(load_or_make(f'./data/dense_gnd_eng_loop_L{L}_h{i}.npy',
                                                 dense_evals, H_loop, temp_hspace)[0])
        plt.plot(HSPACE, gnd_eng_open, label=rf'$L={L}$', color=f'C{C}')
        plt.plot(HSPACE, gnd_eng_loop, color=f'C{C}', linestyle='dotted')
    plt.legend()


def p4_2(Lspace=LSPACE):
    plt.figure()
    for C, L in enumerate(Lspace):
        H = load_or_make_pkl(f'./data/sparse_H_L{L}.pkl', make_sparse_H, L)
        data = load_or_make(f'./data/sparse_gnd_eng_open_L{L}.npz', sparse_eigs, H)
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
