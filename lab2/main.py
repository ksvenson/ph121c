import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import os

import utility
from lab1 import main as l1main

PICS_DIR = './pics/'
COMPRESS_DIR = './compress_pics/'
CACHE_DIR = './data/'
FIGS_DIR = './figs/'
EIGS_DIR = '../lab1/data/'

HSPACE = np.linspace(0, 2, 17)
LSPACE = np.arange(5, 21)

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}


@utility.cache('npz', CACHE_DIR + 'svd')
def get_svd(arr, note=None):
    U, s, Vh = np.linalg.svd(arr, full_matrices=False)
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


def get_entropy(states, l, note=None):
    L = int(np.log2(states.shape[-1]))
    if len(states.shape) == 2:
        M = states.reshape(states.shape[0], 2**l, 2**(L-l))
    elif len(states.shape) == 1:
        M = states.reshape(2**l, 2**(L-l))
    svd = get_svd(M, note=note)
    probs = svd['s']**2
    return -np.sum(probs * np.log(probs), axis=-1)


def entropy_fit(L):
    def fit(l, scale, offset):
        return (scale / 3) * np.log((L / np.pi) * np.sin(np.pi * l / L)) + offset
    return fit


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
        # axes[i].set_yscale('log')
    fig.savefig(FIGS_DIR + 'frobenius.png')


def p5_2():
    phase_h = {
        'Ferromagnet': np.where(HSPACE == 0.25)[0][0],
        'Critical Point': np.where(HSPACE == 1)[0][0],
        'Paramagnet': np.where(HSPACE == 1.75)[0][0]
    }
    fig = plt.figure(figsize=(15, 10))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    entropy = {}
    for bdry_idx, bdry in enumerate(('open', 'loop')):
        entropy[bdry] = {}
        axes = subfigs[bdry_idx].subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        for L in LSPACE:
            data = np.load(EIGS_DIR + f'sparse_eigs_{bdry}_L{L}.npz')
            entropy[bdry][L] = {}
            for i, phase in enumerate(phase_h):
                entropy[bdry][L][phase] = []
                gnd_states = data['evecs'][:, :, 0]
                for l in np.arange(1, L):
                    print(f'getting entropy {bdry} L={L} l={l}')
                    entropy[bdry][L][phase].append(get_entropy(gnd_states, l, note=f'gnd_state_{bdry}_L{L}_l{l}')[phase_h[phase]])

                axes[i].plot(np.arange(1, L), entropy[bdry][L][phase], label=rf'$L={L}$')
                axes[i].set_ylim([0, 1])
                if bdry_idx == 0:
                    axes[i].set_title(phase + rf': $h/J={HSPACE[phase_h[phase]]}$')
                if bdry_idx == 1:
                    axes[i].set_xlabel(r'$\ell$')
                handles, labels = axes[0].get_legend_handles_labels()
        axes[0].set_ylabel('Entanglement Entropy')

    subfigs[0].suptitle(f'Open Boundary')
    subfigs[1].suptitle(f'Periodic Boundary')
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'entropy.png', **FIG_SAVE_OPTIONS)

    summary = {}
    for bdry in entropy:
        summary[bdry] = {}
        for phase in phase_h:
            summary[bdry][phase] = []
            for L in entropy[bdry]:
                summary[bdry][phase].append(entropy[bdry][L][phase][L // 2])

    fig = plt.figure(figsize=(15, 10))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    for bdry_idx, bdry in enumerate(summary):
        axes = subfigs[bdry_idx].subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        for i, phase in enumerate(summary[bdry]):
            axes[i].plot(LSPACE, summary[bdry][phase])
            if bdry_idx == 0:
                axes[i].set_title(phase + rf': $h/J={HSPACE[phase_h[phase]]}$')
            if bdry_idx == 1:
                axes[i].set_xlabel(r'$L$')
        axes[0].set_ylabel(r'$S(L/2, L)$')

    subfigs[0].suptitle(f'Open Boundary')
    subfigs[1].suptitle(f'Periodic Boundary')
    fig.savefig(FIGS_DIR + 'entropy_summary.png', **FIG_SAVE_OPTIONS)

    L_max = LSPACE[-1]
    fit_data = entropy['loop'][L_max]['Critical Point']
    popt, pcov = sp.optimize.curve_fit(entropy_fit(L_max), np.arange(1, L_max), fit_data, p0=(1, 0))
    std = np.sqrt(np.diag(pcov))
    with open('p5_2_fit_results.txt', 'w') as f:
        print(f'scale: {round(popt[0], 5)} \pm {round(std[0], 3)}', file=f)
        print(f'offset: {round(popt[1], 5)} \pm {round(std[1], 3)}', file=f)
    plt.figure()
    plt.plot(np.arange(1, L_max), fit_data, label='Measured E.E.')
    plt.plot(np.arange(1, L_max), entropy_fit(L_max)(np.arange(1, L_max), *popt), label='Equation 26 Fit')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Entanglement Entropy')
    plt.legend()
    plt.savefig(FIGS_DIR + 'entropy_fit.png', **FIG_SAVE_OPTIONS)

    phase = 'Critical Point'
    bdry = 'loop'
    h_idx = phase_h[phase]
    summary = []
    plt.figure()
    for L in LSPACE:
        H = l1main.make_sparse_H(L, hspace=HSPACE, note=f'L{L}')[bdry][h_idx]
        evals, evecs = sp.sparse.linalg.eigsh(H, k=1, which='LA')
        state = evecs.flatten()
        entropy = []
        for l in np.arange(1, L):
            entropy.append(get_entropy(state, l=l, note=f'ext_state_loop_{phase}_L{L}_l{l}'))
        summary.append(entropy[L // 2])
        plt.plot(np.arange(1, L), entropy, label=rf'$L={L}$')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Entanglement Entropy')
    plt.legend(**LEGEND_OPTIONS)
    plt.savefig(FIGS_DIR + f'ext_{bdry}_{phase}_entropy.png', **FIG_SAVE_OPTIONS)

    plt.figure()
    plt.plot(LSPACE, summary)
    plt.xlabel(r'$\ell$')
    plt.ylabel('Entanglement Entropy')
    plt.savefig(FIGS_DIR + f'ext_{bdry}_{phase}_entropy_summary.png', **FIG_SAVE_OPTIONS)


if __name__ == '__main__':
    # p5_1()

    p5_2()
