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
MPS_CACHE_DIR = '.MPS_data/'
FIGS_DIR = './figs/'
EIGS_DIR = '../lab1/data/'

HSPACE = np.linspace(0, 2, 17)
LSPACE = np.arange(5, 17)
DENSE_LSPACE = np.arange(8, 13, 2)

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}

PHASE_H = {
    'Ferromagnet': np.where(HSPACE == 0.25)[0][0],
    'Critical Point': np.where(HSPACE == 1)[0][0],
    'Paramagnet': np.where(HSPACE == 1.75)[0][0]
}


@utility.cache('npz', CACHE_DIR + 'svd')
def get_svd(arr, note=None):
    U, s, Vh = np.linalg.svd(arr, full_matrices=False)
    return {
        'U': U,
        's': s,
        'Vh': Vh
    }


def svd_compress(arr, rank, return_svd=False, svd_idx=None, note=None):
    svd = get_svd(arr, note=note)
    if svd_idx is not None:
        for key in svd:
            svd[key] = svd[key][svd_idx]
    approx_s = []
    for r in rank:
        trun = np.zeros(svd['s'].shape)
        trun[:r] = svd['s'][:r]
        approx_s.append(np.diag(trun))
    approx_s = np.asarray(approx_s)
    if not return_svd:
        return svd['U'] @ approx_s @ svd['Vh']
    else:
        return svd['U'], svd['s'], svd['Vh'], approx_s


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


def MPS_helper(arr, k, A_counter, L, output):
    if len(arr.shape) == 1:
        M = arr.reshape(2, 2**(L-1))
    else:
        M = arr.reshape(arr.shape[0]*2, arr.shape[1] // 2)
    U, s, Vh = np.linalg.svd(M, full_matrices=False)

    A = U[:, :min(k, U.shape[1])]
    W = s[:min(k, s.shape[0]), np.newaxis] * Vh[:min(k, Vh.shape[0])]
    output.append(A)
    if A_counter == L-1:
        output.append(W)
    else:
        MPS_helper(W, k, A_counter+1, L, output)


def make_MPS(state, k, L):
    output = []
    MPS_helper(state, k, 1, L, output)
    return output


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
    fig.savefig(FIGS_DIR + 'p5_1_frobenius.png', **FIG_SAVE_OPTIONS)


def p5_2():
    fig = plt.figure(figsize=(15, 10))
    subfigs = fig.subfigures(nrows=2, ncols=1)
    entropy = {}
    for bdry_idx, bdry in enumerate(('open', 'loop')):
        entropy[bdry] = {}
        axes = subfigs[bdry_idx].subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        for L in LSPACE:
            data = np.load(EIGS_DIR + f'sparse_eigs_{bdry}_L{L}.npz')
            gnd_states = data['evecs'][:, :, 0]
            entropy[bdry][L] = {}
            for i, phase in enumerate(PHASE_H):
                entropy[bdry][L][phase] = []
                for l in np.arange(1, L):
                    print(f'getting entropy {bdry} L={L} l={l}')
                    entropy[bdry][L][phase].append(get_entropy(gnd_states, l, note=f'gnd_state_{bdry}_L{L}_l{l}')[PHASE_H[phase]])

                axes[i].plot(np.arange(1, L), entropy[bdry][L][phase], label=rf'$L={L}$')
                axes[i].set_ylim([0, 1])
                if bdry_idx == 0:
                    axes[i].set_title(phase + rf': $h/J={HSPACE[PHASE_H[phase]]}$')
                if bdry_idx == 1:
                    axes[i].set_xlabel(r'$\ell$')
                handles, labels = axes[0].get_legend_handles_labels()
        axes[0].set_ylabel('Entanglement Entropy')

    subfigs[0].suptitle(f'Open Boundary')
    subfigs[1].suptitle(f'Periodic Boundary')
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'p5_2_entropy.png', **FIG_SAVE_OPTIONS)

    summary = {}
    for bdry in entropy:
        summary[bdry] = {}
        for phase in PHASE_H:
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
                axes[i].set_title(phase + rf': $h/J={HSPACE[PHASE_H[phase]]}$')
            if bdry_idx == 1:
                axes[i].set_xlabel(r'$L$')
        axes[0].set_ylabel(r'$S(L/2, L)$')

    subfigs[0].suptitle(f'Open Boundary')
    subfigs[1].suptitle(f'Periodic Boundary')
    fig.savefig(FIGS_DIR + 'p5_2_entropy_summary.png', **FIG_SAVE_OPTIONS)

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
    plt.savefig(FIGS_DIR + 'p5_2_entropy_fit.png', **FIG_SAVE_OPTIONS)

    phase = 'Ferromagnet'
    bdry = 'loop'
    h_idx = PHASE_H[phase]
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
    plt.savefig(FIGS_DIR + f'p5_2_ext_{bdry}_{phase}_entropy.png', **FIG_SAVE_OPTIONS)

    plt.figure()
    plt.plot(LSPACE, summary)
    plt.xlabel(r'$L$')
    plt.ylabel('Entanglement Entropy')
    plt.savefig(FIGS_DIR + f'p5_2_ext_{bdry}_{phase}_entropy_summary.png', **FIG_SAVE_OPTIONS)


def p5_3():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for L in LSPACE:
        data = np.load(EIGS_DIR + f'sparse_eigs_open_L{L}.npz')
        gnd_states = data['evecs'][:, :, 0]
        evals = data['evals'][:, 0]
        rank = np.arange(1, 2**(L//2) + 1)
        all_ham = l1main.make_sparse_H(L, note=f'L{L}')['open']
        for phase_idx, phase in enumerate(PHASE_H):
            state = gnd_states[PHASE_H[phase]]
            ham = all_ham[PHASE_H[phase]]
            M = state.reshape(2**(L//2), 2**(L - (L//2)))
            U, s, Vh, trun = svd_compress(M, rank, return_svd=True, svd_idx=PHASE_H[phase], note=f'gnd_state_open_L{L}_l{L//2}')
            trun_vals = np.diagonal(trun, axis1=1, axis2=2).T
            trun_M = U @ trun @ Vh
            dk = np.linalg.norm(trun_M - M, axis=(1, 2))
            renorm = np.sum(trun_vals**2, axis=0)

            uv_states = np.array([np.kron(U[:, idx], Vh[idx].conj()) for idx in range(len(s))]).T

            trun_states = uv_states @ trun_vals
            trun_eng = np.diag(trun_states.conj().T @ ham @ trun_states) / renorm

            axes[phase_idx].plot(dk, np.abs(evals[PHASE_H[phase]] - trun_eng), label=rf'$L={L}$')
            axes[phase_idx].set_title(phase + rf': $h/J={HSPACE[PHASE_H[phase]]}$')
            axes[phase_idx].set_xlabel(rf'$d(k)$')

    axes[0].set_ylabel('$\Delta E$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'p5_3_schmidt_decomp.png', **FIG_SAVE_OPTIONS)


def p5_4():
    phase = 'Paramagnet'
    bdry = 'loop'
    degen_margin = 1e-5
    num_states = 3

    fig, axes = plt.subplots(1, num_states, sharex=True, sharey=True, figsize=(5 * num_states, 5))
    engs = []
    for L in DENSE_LSPACE:
        eigs = np.load(CACHE_DIR + f'dense_eigs_{bdry}_L{L}.npz')
        evecs = eigs['evecs'][PHASE_H[phase]]
        evals = eigs['evals'][PHASE_H[phase]]

        sort_idx = np.argsort(evals)
        evals = np.real(evals[sort_idx])
        evecs = evecs[:, sort_idx]

        all_mid_idx = np.argsort(np.abs(evals))
        mid_idx = []
        for idx in all_mid_idx:
            if np.abs(evals[idx] - evals[idx + 1]) > degen_margin and np.abs(evals[idx] - evals[idx - 1]) > degen_margin:
                mid_idx.append(idx)
            if len(mid_idx) >= num_states:
                break
        mid_idx = np.array(mid_idx)
        mid_idx_sort = np.argsort(evals[mid_idx])
        mid_idx = mid_idx[mid_idx_sort]
        engs.append(evals[mid_idx])
        states = evecs[:, mid_idx].T

        entropy = []
        for l in np.arange(1, L):
            entropy.append(get_entropy(states, l=l, note=f'mid_state_{bdry}_{phase}_L{L}_l{l}'))
        entropy = np.array(entropy).T

        for j, ax in enumerate(axes):
            ax.plot(np.arange(1, L), entropy[j], label=rf'$L={L}$')
            ax.set_xlabel(r'$\ell$')

    engs = np.array(engs)
    for i, ax in enumerate(axes):
        ax.set_title(r'$E_\text{avg}=$' + rf'${round(np.mean(engs, axis=0)[i], 3)}$')

    axes[0].set_ylabel('Entanglement Entropy')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + f'p5_4_{phase}_{bdry}_mid_state_entropy.png', **FIG_SAVE_OPTIONS)


def p5_5():
    MPS_H = {
        'crit': np.where(HSPACE == 1)[0][0],
        'close': np.where(HSPACE == 1.25)[0][0]
    }

    for phase in MPS_H:
        for L in LSPACE:
            eigs = np.load(EIGS_DIR + f'sparse_eigs_open_L{L}.npz')
            evecs = eigs['evecs'][MPS_H[phase]]
            evals = eigs['evals'][MPS_H[phase]]
            gnd_state = evecs[:, 0]
            mps = make_MPS(gnd_state, 3, L)

            print(mps)
            for tensor in mps:
                print(tensor.shape)

            quit()


if __name__ == '__main__':
    # p5_1()

    # p5_2()

    # p5_3()

    # p5_4()

    p5_5()
