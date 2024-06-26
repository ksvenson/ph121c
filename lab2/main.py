import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from PIL import Image
import os

import utility
from lab1 import main as l1main
from lab4 import main as l4main
from typing import Literal


PICS_DIR = './pics/'
COMPRESS_DIR = './compress_pics/'
CACHE_DIR = './data/'
MPS_CACHE_DIR = './MPS_data/'
FIGS_DIR = './figs/'
EIGS_DIR = '../lab1/data/'

HSPACE = np.linspace(0, 2, 17)
LSPACE = np.arange(5, 21)
DENSE_LSPACE = np.arange(8, 13, 2)

LEGEND_OPTIONS = {'bbox_to_anchor': (0.9, 0.5), 'loc': 'center left'}
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}

PHASE_H = {
    'Ferromagnet': np.where(HSPACE == 0.25)[0][0],
    'Critical Point': np.where(HSPACE == 1)[0][0],
    'Paramagnet': np.where(HSPACE == 1.75)[0][0]
}
MPS_H = {
    'crit': np.where(HSPACE == 1)[0][0],
    'close': np.where(HSPACE == 1.25)[0][0]
}

operator_type = Literal['sigz', 'sigx']

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


def reshape_U_to_A(U, k):
    mask = np.arange(U.shape[0]) % 2
    A0 = U[mask == 0, :k]
    A1 = U[mask == 1, :k]
    return np.array([A0, A1])


def MPS_helper(arr, k, A_counter, L, output):
    M = arr.reshape(arr.shape[0]*2, arr.shape[1] // 2)
    U, s, Vh = np.linalg.svd(M, full_matrices=False)

    A = reshape_U_to_A(U, k)
    output.append(A)

    W = s[:k, np.newaxis] * Vh[:k]
    if A_counter == L-1:
        return W
    else:
        return MPS_helper(W, k, A_counter+1, L, output)


@utility.cache('pkl', MPS_CACHE_DIR + 'mps')
def make_MPS(state, k, L, note=None):
    output = []
    M = state.reshape(2, 2**(L-1))
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    A1 = U[:, :k]
    W = s[:k, np.newaxis] * Vh[:k]
    AL = MPS_helper(W, k, 2, L, output)
    return A1, output, AL


@utility.cache('npy', MPS_CACHE_DIR + 'mps_state')
def virtual_contract(A1, A, AL, L, note=None):
    # ith spin is given by the ith bit in `L`, left to right
    output_state = []
    for state in range(2**L):
        output = A1[(state >> (L-1)) & 1]
        for i, mat in enumerate(A):
            output = output @ mat[(state >> (L-i-2)) & 1]
        output_state.append(output @ AL[:, state & 1])
    return np.array(output_state)


def mps_contract_helper(A1, A, A_count, spin_idx=None, operator: operator_type=None):
    if A_count == 1:
        A1_star = A1.conj()
        if operator == 'sigx' and spin_idx == 0:
            A1_star = A1_star[::-1]
        prod = np.array([np.outer(A1_star[i], A1[i]) for i in range(2)])
        if operator == 'sigz' and spin_idx == 0:
            prod[1] *= -1
    else:
        next_A = A[A_count - 2]
        next_A_dag = np.transpose(next_A, (0, 2, 1)).conj()
        if operator == 'sigx' and spin_idx == A_count - 1:
            next_A_dag = next_A_dag[::-1]
        prod = next_A_dag @ mps_contract_helper(A1, A, A_count-1, spin_idx=spin_idx, operator=operator) @ next_A
        if operator == 'sigz' and (spin_idx == A_count - 1 or spin_idx == A_count - 2):
            prod[1] *= -1
    return np.sum(prod, axis=0)


def mps_norm(A1, A, AL, L):
    result = mps_contract_helper(A1, A, L-1)
    prod = np.array([np.sum(col.conj() * (result @ col)) for col in AL.T])
    return np.sum(prod)


def mps_sigz(A1, A, AL, L, spin_idx):
    result = mps_contract_helper(A1, A, L-1, spin_idx=spin_idx, operator='sigz')
    prod = np.array([np.sum(col.conj() * (result @ col)) for col in AL.T])
    if spin_idx == L-2:
        prod[1] *= -1
    return np.sum(prod)


def mps_sigx(A1, A, AL, L, spin_idx):
    result = mps_contract_helper(A1, A, L-1, spin_idx=spin_idx, operator='sigx')
    AL_dag = AL.T.conj()
    if spin_idx == L-1:
        AL_dag = AL_dag[::-1]
    prod = np.array([np.sum(AL_dag[i] * (result @ AL[:, i])) for i in range(2)])
    return np.sum(prod)


def mps_eng(A1, A, AL, L, h_J):
    total = 0
    for spin_idx in range(L-1):
        total += -1 * mps_sigz(A1, A, AL, L, spin_idx)
        total += -1 * h_J * mps_sigx(A1, A, AL, L, spin_idx)
    total += -1 * h_J * mps_sigx(A1, A, AL, L, L-1)
    return total / mps_norm(A1, A, AL, L)


def compute_s_vals(L, A1, A, AL, num=2):
    A_list = [A1]
    for _ in range(L - 2):
        A_list.append(A.copy())
    A_list.append(AL)

    mps = l4main.MPS(L, A_list, 0, 0, 0)
    mps.renormalize()
    mps.restore_canonical()
    mps.move_ortho_center(0)
    state = mps.get_state()
    brute_s_vals = []
    mps_s_vals = []
    for l in range(1, L):
        M = state.reshape(2**l, 2**(L - l))
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        brute_s_vals.append(s[:num])

        mps_s_vals.append(mps.get_schmidt_vals()[:num])
        mps.shift_ortho_center(left=False)

    return brute_s_vals, mps_s_vals


def print_s_vals(s_vals, fname):
    with open(fname, 'w') as f:
        print(f'First Schmidt Value: {[round(arr[0], 5) for arr in s_vals]}', file=f)
        print(f'Second Schmidt Value: {[round(arr[1], 5) for arr in s_vals]}', file=f)
        print(f'Third Schmidt Value: {[round(arr[2], 5) for arr in s_vals if len(arr) >= 3]}', file=f)


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
        print(f'scale: {popt[0]} \pm {std[0]}', file=f)
        print(f'offset: {popt[1]} \pm {std[1]}', file=f)
    plt.figure()
    plt.plot(np.arange(1, L_max), fit_data, label='Measured E.E.')
    plt.plot(np.arange(1, L_max), entropy_fit(L_max)(np.arange(1, L_max), *popt), label='Equation 26 Fit')
    plt.xlabel(r'$\ell$')
    plt.ylabel('Entanglement Entropy')
    plt.legend()
    plt.savefig(FIGS_DIR + 'p5_2_entropy_fit.png', **FIG_SAVE_OPTIONS)

    phase = 'Paramagnet'
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
    plt.ylabel('$S(L/2, L)$')
    plt.savefig(FIGS_DIR + f'p5_2_ext_{bdry}_{phase}_entropy_summary.png', **FIG_SAVE_OPTIONS)


def p5_3():
    fig_dk, axes_dk = plt.subplots(1, 3, figsize=(15, 5))
    fig_de, axes_de = plt.subplots(1, 3, figsize=(15, 5))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for L in LSPACE:
        print(f'L={L}')
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

            axes_dk[phase_idx].plot(rank, dk, label=rf'$L={L}$')
            axes_dk[phase_idx].set_title(phase + rf': $h/J={HSPACE[PHASE_H[phase]]}$')
            axes_dk[phase_idx].set_xlabel(rf'Rank')
            axes_dk[phase_idx].set_xscale('log')
            axes_dk[phase_idx].set_yscale('log')

            axes_de[phase_idx].plot(rank, np.abs(evals[PHASE_H[phase]] - trun_eng), label=rf'$L={L}$')
            axes_de[phase_idx].set_title(phase + rf': $h/J={HSPACE[PHASE_H[phase]]}$')
            axes_de[phase_idx].set_xlabel(rf'Rank')
            axes_de[phase_idx].set_xscale('log')
            axes_de[phase_idx].set_yscale('log')

            axes[phase_idx].plot(dk, np.abs(evals[PHASE_H[phase]] - trun_eng), label=rf'$L={L}$')
            axes[phase_idx].set_title(phase + rf': $h/J={HSPACE[PHASE_H[phase]]}$')
            axes[phase_idx].set_xlabel(rf'$d(k)$')
            axes[phase_idx].set_xscale('log')
            axes[phase_idx].set_yscale('log')

    axes_dk[0].set_ylabel('$d(k)$')
    axes_de[0].set_ylabel('$\Delta E$')
    axes[0].set_ylabel('$\Delta E$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig_dk.legend(handles, labels, **LEGEND_OPTIONS)
    fig_de.legend(handles, labels, **LEGEND_OPTIONS)
    fig.legend(handles, labels, **LEGEND_OPTIONS)

    fig_dk.savefig(FIGS_DIR + 'p5_3_schmidt_decomp_dk.png', **FIG_SAVE_OPTIONS)
    fig_de.savefig(FIGS_DIR + 'p5_3_schmidt_decomp_de.png', **FIG_SAVE_OPTIONS)
    fig.savefig(FIGS_DIR + 'p5_3_schmidt_decomp.png', **FIG_SAVE_OPTIONS)


def p5_4():
    phase = 'Critical Point'
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
    bdry = 'open'
    keys = ['overlap', 'norm', 'eng', 'eng_err']
    figs = {}
    axes = {}
    for key in keys:
        figs[key], axes[key] = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))

    data = {}
    for phase_idx, phase in enumerate(MPS_H):
        for L in LSPACE:
            eigs = np.load(EIGS_DIR + f'sparse_eigs_{bdry}_L{L}.npz')
            evecs = eigs['evecs'][MPS_H[phase]]
            evals = eigs['evals'][MPS_H[phase]]
            gnd_state = evecs[:, 0]
            gnd_eng = evals[0]
            k_space = 2**np.arange(0, L//2)
            for key in keys:
                data[key] = []
            for k in k_space:
                A1, A, AL = make_MPS(gnd_state, k, L, note=f'{phase}_{bdry}_L{L}_k{k}')
                mps_state = virtual_contract(A1, A, AL, L, note=f'contract_{phase}_{bdry}_L{L}_k{k}')
                data['overlap'].append(np.sum(mps_state.conj() * gnd_state))
                data['norm'].append(mps_norm(A1, A, AL, L))
                data['eng'].append(mps_eng(A1, A, AL, L, HSPACE[MPS_H[phase]]))
                data['eng_err'].append(np.abs(data['eng'][-1] - gnd_eng))
            for key in data:
                axes[key][phase_idx].plot(k_space, data[key], label=rf'$L={L}$')
                axes[key][phase_idx].set_xlabel(r'$k$')
                axes[key][phase_idx].set_xscale('log')
                axes[key][phase_idx].set_yscale('log')
    axes['overlap'][0].set_ylabel(r'Overlap $\langle \tilde{\psi}_\text{gs}(k) | \psi_\text{gs}\rangle$')
    axes['norm'][0].set_ylabel(r'Normalization $\langle \tilde{\psi}_\text{gs}(k) | \tilde{\psi}_\text{gs}(k)\rangle$')
    axes['eng'][0].set_ylabel(r'Energy $\langle \tilde{\psi}_\text{gs}(k) | H |\tilde{\psi}_\text{gs}(k)\rangle / \langle \tilde{\psi}_\text{gs}(k) | \tilde{\psi}_\text{gs}(k)\rangle$')
    axes['eng_err'][0].set_ylabel(r'Energy Error $\Delta E$')

    for key in data:
        title_help = HSPACE[MPS_H['crit']]
        axes[key][0].set_title(rf'Critical Point: $h/J={title_help}$')
        title_help = HSPACE[MPS_H['close']]
        axes[key][1].set_title(rf'$h/J={title_help}$')
        handles, labels = axes[key][0].get_legend_handles_labels()
        figs[key].legend(handles, labels, **LEGEND_OPTIONS)
        figs[key].savefig(FIGS_DIR + f'p5_5_mps_{key}.png', **FIG_SAVE_OPTIONS)


def p5_5_1():
    bdry = 'open'

    fig_eng, axes_eng = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    fig_diff, axes_diff = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    fig_correl, axes_correl = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5))
    for phase_idx, phase in enumerate(MPS_H):
        for color, L in enumerate(LSPACE):
            print(f'L: {L}')
            eigs = np.load(EIGS_DIR + f'sparse_eigs_{bdry}_L{L}.npz')
            gnd_state = eigs['evecs'][MPS_H[phase]][:, 0]
            gnd_eng = eigs['evals'][MPS_H[phase]][0]
            kspace = 2**np.arange(0, L//2)

            eng = []
            for k in kspace:
                mps = l4main.MPS.make_from_state(gnd_state, L, k, hx=HSPACE[MPS_H[phase]], hz=0, ortho_center=0)
                if L == LSPACE[-1]:
                    axes_correl[phase_idx].plot(np.arange(L), mps.correlation(), label=rf'$k={k}$')
                eng.append(mps.measure_energy() / mps.norm())
            eng = np.array(eng)

            axes_eng[phase_idx].plot(kspace, eng, label=rf'$L={L}$', color=f'C{color}')
            axes_eng[phase_idx].axhline(gnd_eng, color=f'C{color}', linestyle='dotted')

            axes_diff[phase_idx].plot(kspace, np.abs(eng - gnd_eng), label=rf'$L={L}$', color=f'C{color}')

        title_help = HSPACE[MPS_H[phase]]
        axes_eng[phase_idx].set_title(rf'$h/J={title_help}$')
        axes_eng[phase_idx].set_xlabel(r'$k$')
        axes_eng[phase_idx].set_xscale('log')

        axes_diff[phase_idx].set_title(rf'$h/J={title_help}$')
        axes_diff[phase_idx].set_xlabel(r'$k$')
        axes_diff[phase_idx].set_xscale('log')
        axes_diff[phase_idx].set_yscale('log')

        axes_correl[phase_idx].set_title(rf'$h/J={title_help}$')
        axes_correl[phase_idx].set_xlabel(r'$r$')

    axes_eng[0].set_ylabel('Ground State Energy')
    handles, labels = axes_eng[0].get_legend_handles_labels()
    fig_eng.legend(handles, labels, **LEGEND_OPTIONS)
    fig_eng.savefig(FIGS_DIR + f'p5_5_1_mps_eng.png', **FIG_SAVE_OPTIONS)

    axes_diff[0].set_ylabel('Ground State Energy Difference')
    handles, labels = axes_diff[0].get_legend_handles_labels()
    fig_diff.legend(handles, labels, **LEGEND_OPTIONS)
    fig_diff.savefig(FIGS_DIR + f'p5_5_1_mps_eng_diff.png', **FIG_SAVE_OPTIONS)

    axes_correl[0].set_ylabel(r'$\langle \sigma_1^z \sigma_{1+r}^z \rangle$')
    handles, labels = axes_correl[0].get_legend_handles_labels()
    fig_correl.legend(handles, labels, **LEGEND_OPTIONS)
    fig_correl.savefig(FIGS_DIR + f'p5_5_1_mps_correl.png', **FIG_SAVE_OPTIONS)


def p5_5_2():
    L = 10
    A1 = (1/np.sqrt(2)) * np.array([1, -1, 1, 1]).reshape(2, 1, 2)
    A = (1/np.sqrt(2)) * np.array([0, 0, 1, -1, 1, 1, 0, 0]).reshape(2, 2, 2)
    AL = np.array([0, 1, 1, 0]).reshape(2, 2, 1)

    brute_s_vals, mps_s_vals = compute_s_vals(L, A1, A, AL)
    print_s_vals(brute_s_vals, 'p5_5_2_svals_first.txt')

    L = 20
    A1 = np.array([2, -1, 1, 2]).reshape(2, 1, 2)
    AL = np.array([3, 1, 1, 3]).reshape(2, 2, 1)

    brute_s_vals, mps_s_vals = compute_s_vals(L, A1, A, AL, num=2)
    brute_s_vals = np.array(brute_s_vals)
    mps_s_vals = np.array(mps_s_vals)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(1 + np.arange(len(brute_s_vals)), brute_s_vals[:, 0], label='First Schmidt Value')
    axes[0].plot(1 + np.arange(len(brute_s_vals)), brute_s_vals[:, 1], label='Second Schmidt Value')
    axes[0].set_title('Brute Force Schmidt Values')
    axes[0].set_xlabel(r'$\ell$')

    axes[1].plot(1 + np.arange(len(mps_s_vals)), mps_s_vals[:, 0], label='First Schmidt Value')
    axes[1].plot(1 + np.arange(len(mps_s_vals)), mps_s_vals[:, 1], label='Second Schmidt Value')
    axes[1].set_title('MPS Schmidt Values')
    axes[1].set_xlabel(r'$\ell$')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'p5_5_2_s_vals.png', **FIG_SAVE_OPTIONS)

    fig, axes = plt.subplots(figsize=(5, 5))
    axes.plot(1 + np.arange(len(brute_s_vals)), np.abs(brute_s_vals[:, 0] - mps_s_vals[:, 0]), label='First Schmidt Value')
    axes.plot(1 + np.arange(len(brute_s_vals)), np.abs(brute_s_vals[:, 1] - mps_s_vals[:, 1]), label='Second Schmidt Value')
    axes.set_xlabel(r'$\ell$')
    axes.set_ylabel('Difference')
    axes.set_yscale('log')
    fig.legend(**LEGEND_OPTIONS)
    fig.savefig(FIGS_DIR + 'p5_5_2_s_vals_diff.png', **FIG_SAVE_OPTIONS)


if __name__ == '__main__':
    p5_1()

    p5_2()

    p5_3()

    p5_4()

    p5_5()

    p5_5_1()

    p5_5_2()
