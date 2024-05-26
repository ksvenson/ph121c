import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import utility

import time

FIG_DIR = './figs/'
CACHE_DIR = './data/'

FIELD_VALS = {'hx': -1.05, 'hz': 0.5}
TOL = 1e-2


test_hx = FIELD_VALS['hx']
test_hz = FIELD_VALS['hz']
test_dt = 0.01
test_k = 16
def measure(L):
    mps = MPS.make_initial_state(L, test_hx, test_hz)
    mps.evolve(test_dt, test_k)
    mps.renormalize()
    mps.measure_energy(k=test_k)

def test_cool(L):
    mps = MPS.make_initial_state(L, test_hx, test_hz)
    return mps.cool(test_dt, test_k)



@utility.cache('npy', CACHE_DIR + 'l4_dense_H')
def make_dense_H(L, hx, hz, note=None):
    H = np.zeros((2**L, 2**L))
    for i in range(2**L):
        # Perform a XOR between states, and states shifted by 1 bit. Make sure that the first bit is zero.
        H[i, i] += 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)

        # h_z field
        H[i, i] += -1 * hz * (2 * i.bit_count() - L)

        # h_x field
        flips = np.arange(L)
        H[i ^ (1 << flips), i] += -1 * hx
    return H


@utility.cache('npz', CACHE_DIR + 'l4_dense_eigs')
def dense_eigs(L, hx, hz, note=None):
    print(f'finding dense evals: L={L}')
    H = make_dense_H(L, hx, hz, note=note)
    evals, evecs = sp.linalg.eigh(H)
    return {'evals': evals, 'evecs': evecs}


class MPS:
    """
    Represents an MPS state.
    """
    def __init__(self, L, A_list, ortho_center, hx, hz):
        """
        Creates an MPS given the list of A tensors.
        :param L: System size.
        :param A_list: List of A^j tensors.
        :param ortho_center: Index of orthogonality center.
        """
        self.L = L
        self.A_list = A_list
        self.ortho_center = ortho_center
        self.hx = hx
        self.hz = hz

        # Slightly modified matrix from Equation 20.
        # This is because we work in a basis were the zero index corresponds to spin down.
        self.field_op = np.array([
            [self.hz, -self.hx],
            [-self.hx, -self.hz]
        ])

        # Helper matrix for two-site operations
        self.xor = np.array([
            [-1, 1],
            [1, -1]
        ])

    @staticmethod
    def __svd_W(W, k=None, left=True):
        W = np.moveaxis(W, (0, 1, 2, 3), (1, 2, 0, 3))
        W = W.reshape(W.shape[0] * W.shape[1], W.shape[2] * W.shape[3])

        U, s, Vh = np.linalg.svd(W, full_matrices=False)
        if k is not None:
            U = U[:, :k]
            s = s[:k]
            Vh = Vh[:k]
        if left:
            U *= s
        else:
            Vh = s[:, np.newaxis] * Vh
        split_idx = Vh.shape[1] // 2
        return np.stack((U[::2], U[1::2])), np.stack((Vh[:, :split_idx], Vh[:, split_idx:]))

    @classmethod
    def make_from_state(cls, state, L, k, hx, hz, ortho_center=None):
        """
        Returns an MPS representing `state` in the sigma z basis.
        :param state: State in the sigma z basis.
        :param k: Bond dimension.
        :param L: System size.
        :param hx: x field parameter.
        :param hz: z field parameter.
        :param ortho_center: Desired location of the orthonormality center.
        :return: MPS object representing `state`.
        """
        A_list = []
        W = state.reshape(1, 2**L)
        cls.__make_from_state_helper(W, k, 1, L, A_list)
        mps = cls(L, A_list, L-1, hx, hz)
        if ortho_center is not None:
            for _ in range(L - ortho_center - 1):
                mps.shift_ortho_center(k=k, left=True)
        return mps

    @classmethod
    def __make_from_state_helper(cls, W, k, A_counter, L, A_list):
        """
        Helper method for `make_from_state`. Appends A^{`A_counter`} to the end of `A_list`.
        :param W: Next W tensor to perform an SVD on.
        :param k: Bond dimension
        :param A_counter: See general function description above.
        :param L: System size.
        :param A_list: Growing list for the A^j tensors.
        :return: A^L.
        """
        M = W.reshape(W.shape[0] * 2, W.shape[1] // 2)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)

        # Even indices of `U` become (A^j)^0, and odd indices (A^j)^1
        A = np.stack((U[::2, :k], U[1::2, :k]))
        A_list.append(A)

        W = s[:k, np.newaxis] * Vh[:k]
        if A_counter == L - 1:
            A_list.append(np.stack((W[:, :1], W[:, 1:])))
        else:
            cls.__make_from_state_helper(W, k, A_counter + 1, L, A_list)

    @classmethod
    def make_initial_state(cls, L, hx, hz, neel=False):
        A_list = []
        for i in range(L):
            if neel and i % 2 == 1:
                A_list.append(np.arange(2)[::-1].reshape(2, 1, 1))
            else:
                A_list.append(np.arange(2).reshape(2, 1, 1))
        return cls(L, A_list, 0, hx, hz)

    def norm(self):
        A = self.A_list[0]
        A_dag = np.transpose(A, axes=(0, 2, 1)).conj()
        contract = np.einsum('abc,acd->bd', A_dag, A)
        for j in range(1, self.L):
            A = self.A_list[j]
            A_dag = np.transpose(A, axes=(0, 2, 1)).conj()
            contract = np.einsum('abc,cd,ade->be', A_dag, contract, A)
        return np.trace(contract)

    def renormalize(self):
        # We renormalize at the orthogonality center so that we maintain the canonical form
        self.A_list[self.ortho_center] /= np.sqrt(self.norm())

    def __apply_field(self, dt):
        mag = np.sqrt(self.hx**2 + self.hz**2)
        phi = dt * mag
        n_dot_sigma = self.field_op / mag
        field = np.cosh(phi) * np.identity(2) - np.sinh(phi) * n_dot_sigma
        for idx, A in enumerate(self.A_list):
            self.A_list[idx] = np.tensordot(field, A, axes=([1], [0]))

    def __apply_two_site(self, dt, parity):
        exp_xor = np.exp(-1 * dt * self.xor)
        for j in range(parity, self.L-1, 2):
            # multiply A^j and A^{j+1} together
            W = np.einsum('abc,dce->adbe', self.A_list[j], self.A_list[j+1])
            # multiply by the 2-site operator
            W = np.einsum('ab,abcd->abcd', exp_xor, W)
            # W = np.einsum('ab,abcd->bcad', exp_xor, W)
            self.A_list[j], self.A_list[j+1] = self.__class__.__svd_W(W, left=False)

    def __restore_canonical(self, k):
        self.ortho_center = 0
        self.sweep(k=None, left=False)
        # self.sweep(k=k, left=True)

    def shift_ortho_center(self, k=None, left=True):
        if left:
            assert self.ortho_center != 0
            l_idx = self.ortho_center - 1
            r_idx = self.ortho_center
        else:
            assert self.ortho_center != self.L - 1
            l_idx = self.ortho_center
            r_idx = self.ortho_center + 1

        W = np.einsum('abc,dce->adbe', self.A_list[l_idx], self.A_list[r_idx])
        # W = np.einsum('abc,dce->dbae', self.A_list[l_idx], self.A_list[r_idx])
        self.A_list[l_idx], self.A_list[r_idx] = self.__class__.__svd_W(W, k=k, left=left)

        if left:
            self.ortho_center -= 1
        else:
            self.ortho_center += 1

    def sweep(self, k=None, left=True):
        if left:
            assert self.ortho_center == self.L-1
        else:
            assert self.ortho_center == 0
        for _ in range(self.L-1):
            self.shift_ortho_center(k=k, left=left)

    def evolve(self, dt, k):
        self.__apply_field(dt)
        self.__apply_two_site(dt, 0)
        self.__apply_two_site(dt, 1)
        self.__restore_canonical(k)

    def measure_energy(self, k=None):
        assert self.ortho_center == 0 or self.ortho_center == self.L - 1
        left = self.ortho_center == self.L - 1
        measure_order = np.arange(self.L)
        if left:
            measure_order = measure_order[::-1]

        energy = 0
        for j in measure_order:
            # Compute the field energy
            A = self.A_list[j]
            A_dag = np.transpose(A, axes=(0, 2, 1)).conj()
            traces = np.einsum('abc,dcb->ad', A_dag, A)
            energy += np.sum(self.field_op * traces)

            if left and j == 0:
                break
            if (not left) and j == self.L - 1:
                break

            # Compute the two-site energy
            if left:
                next_A = A
                next_A_dag = A_dag
                A = self.A_list[j-1]
                A_dag = np.transpose(A, axes=(0, 2, 1)).conj()
            else:
                next_A = self.A_list[j+1]
                next_A_dag = np.transpose(next_A, axes=(0, 2, 1)).conj()

            A_prod = np.einsum('abc,ace->abe', A_dag, A)
            next_A_prod = np.einsum('abc,ace->abe', next_A, next_A_dag)

            for next_idx in range(2):
                for idx in range(2):
                    energy += self.xor[next_idx, idx] * np.einsum('ab,ba->', next_A_prod[next_idx], A_prod[idx])

            # Move the orthogonality center to the next site:
            self.shift_ortho_center(left=left, k=k)
        return energy

    def cool(self, dt, k):
        energy = [self.measure_energy(k=k)]
        runtime = []
        while True:
            start = time.time()
            for A in self.A_list:
                assert max(A.shape) <= k
            self.evolve(dt, k)
            self.renormalize()
            energy.append(self.measure_energy(k=k))

            runtime.append(time.time() - start)

            if np.abs((energy[-1] - energy[-2]) / energy[-1]) < TOL:
                break
        energy = np.array(energy)

        plt.plot(runtime, label=self.L)

        return energy, np.mean(runtime)


def p4_1_fix_L(dtspace, L=12, k=16, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz']):
    eigs = dense_eigs(L, hx, hz, note=f'L{L}_hx{hx}_hz{hz}')
    gnd_eng = eigs['evals'][0]

    state = np.zeros(2**L)
    state[-1] = 1

    fig, axes = plt.subplots(figsize=(5, 5))
    for dt in dtspace:
        print(f'dt: {round(dt, 5)}')
        mps = MPS.make_initial_state(L, hx, hz)
        energy = mps.cool(dt, k)

        tspace = dt * np.arange(0, len(energy))
        axes.plot(tspace, energy, label=rf'$\delta \tau = {round(dt, 5)}$')
        print(f'final energy: {energy[-1]}')
    print(f'true energy: {gnd_eng}')

    axes.axhline(gnd_eng, label='Ground State Energy', color='black', linestyle='dotted')
    axes.set_xlabel(r'Imaginary Time $\delta \tau$')
    axes.set_ylabel('Energy')

    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_gnd_eng_convergence.png', **utility.FIG_SAVE_OPTIONS)


def p4_1_fix_dt(Lspace, dt=0.01, k=16, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz']):
    fig, axes = plt.subplots(figsize=(5, 5))
    gnd_eng = []
    for L in Lspace:
        print(f'L: {L}')
        mps = MPS.make_initial_state(L, hx, hz)
        gnd_eng.append(mps.cool(dt, k)[-1])
    gnd_eng = np.array(gnd_eng)

    bulk_eng = []
    for i in range(len(Lspace) - 1):
        bulk_eng.append((gnd_eng[i + 1] - gnd_eng[i]) / (Lspace[i+1] - Lspace[i]))
    bulk_eng = np.array(bulk_eng)

    L_step = Lspace[1] - Lspace[0]
    axes.plot(Lspace, gnd_eng / Lspace, label=r'$E_0(L)$')
    axes.plot(Lspace[:-1], bulk_eng, label=rf'$(E_0(L + {L_step}) - E_0(L)) / {L_step}$')
    axes.set_xlabel(r'$L$')
    axes.set_ylabel('Energy per Site')
    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_L_infty.png', **utility.FIG_SAVE_OPTIONS)


if __name__ == '__main__':
    DTSPACE = 0.1**np.arange(1, 3)
    LSPACE = np.arange(32, 256, 10)
    # LSPACE = np.arange(12, 15)

    # p4_1_fix_L(DTSPACE)

    # p4_1_fix_dt(LSPACE)

    plt.figure()
    mean_runtime_list = []
    for L in np.arange(5, 50, 5):
        print(L)
        energy, mean_runtime = test_cool(L)
        mean_runtime_list.append(mean_runtime)
    plt.legend()

    plt.figure()
    plt.plot(np.arange(5, 50, 5), mean_runtime_list)

    plt.show()
