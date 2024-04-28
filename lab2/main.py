import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import os

import utility

PICS_DIR = './pics/'
COMPRESS_DIR = './compress_pics/'
CACHE_DIR = './data/'
FIGS_DIR = './figs/'

HSPACE = np.linspace(0, 2, 17)


@utility.cache('npz', CACHE_DIR + 'svd')
def get_svd(arr, note=None):
    U, s, Vh = sp.linalg.svd(arr, full_matrices=False)
    return {
        'U': U,
        's': s,
        'Vh': Vh
    }


def svd_compress(arr, rank, note=None):
    svd = get_svd(arr, note=note)
    approx_s = []
    for r in rank:
        trun = np.zeros(svd['s'].shape)
        trun[:r] = svd['s'][:r]
        approx_s.append(np.diag(trun))
    approx_s = np.asarray(approx_s)
    return svd['U'] @ approx_s @ svd['Vh']


def p5_1():
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    for i, filename in enumerate(os.listdir(PICS_DIR)):
        name, ext = os.path.splitext(filename)
        img = Image.open(PICS_DIR + filename).convert('L')
        arr = np.asarray(img)

        rank = np.min(arr.shape) // 2**np.arange(10)
        compress = svd_compress(arr, rank, note=name)
        norm = np.linalg.norm(compress - arr, axis=(1, 2))

        with open(f'p5_1_{name}_norms.txt', 'w') as norm_file:
            for j, comp_arr in enumerate(compress):
                print(name + f' rank {rank[j]}: {np.linalg.norm(comp_arr - arr)}', file=norm_file)
                gray_img = Image.fromarray(comp_arr).convert('L')
                gray_img.save(COMPRESS_DIR + name + f'_rank{rank[j]}.png')

        axes[i].plot(rank, norm)
        axes[i].set_title(name.capitalize())
        axes[i].set_xlabel('Rank')
        axes[i].set_xscale('log')
        if i == 0:
            axes[i].set_ylabel('Frobenius Norm')
        axes[i].set_yscale('log')
    fig.savefig(FIGS_DIR + 'frobenius.png')


def p5_2():
    h = {
        'Ferromagnet': np.where(HSPACE == 0.25)[0][0],
        'Critical Point': np.where(HSPACE == 1)[0][0],
        'Paramagnet': np.where(HSPACE == 1.75)[0][0]
    }
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))




if __name__ == '__main__':
    # p5_1()

    p5_2()
