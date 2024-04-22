import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import os

import utility

PICS_DIR = './pics/'
COMPRESS_DIR = './compress_pics/'
CACHE_DIR = './data/'


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
    for filename in os.listdir(PICS_DIR):
        name, ext = os.path.splitext(filename)
        img = Image.open(PICS_DIR + filename).convert('L')
        arr = np.asarray(img)

        rank = np.arange(3)
        compress = svd_compress(arr, rank, note=name)

        for i, arr in enumerate(compress):
            gray_img = Image.fromarray(arr).convert('L')
            gray_img.save(COMPRESS_DIR + name + f'_rank{rank[i]}.png')


if __name__ == '__main__':
    p5_1()
