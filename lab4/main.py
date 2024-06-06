import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import utility

FIG_DIR = './figs/'
CACHE_DIR = './data/'

FIELD_VALS = {'hx': -1.05, 'hz': 0.5}
TOL = 1e-3


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
    def __svd_W(W, k=None, left=True, return_s=False):
        W = np.moveaxis(W, (0, 1, 2, 3), (1, 2, 0, 3))
        W = W.reshape(W.shape[0] * W.shape[1], W.shape[2] * W.shape[3])

        try:
            U, s, Vh = np.linalg.svd(W, full_matrices=False)
        except np.linalg.LinAlgError as err:
            print(err)
            print(f'nan: {np.sum(np.isnan(W))}')
            np.save('W_err.npy', W)
            quit()

        if k is not None:
            U = U[:, :k]
            s = s[:k]
            Vh = Vh[:k]
        if return_s:
            return s
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
            mps.move_ortho_center(ortho_center)
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
    def make_isotropic_state(cls, L, state, hx, hz):
        A_list = [state.reshape(2, 1, 1) for _ in range(L)]
        return cls(L, A_list, 0, hx, hz)

    @classmethod
    def make_neel_state(cls, L, hx, hz):
        A_list = []
        for i in range(L):
            if i % 2 == 0:
                A_list.append(np.array([0, 1]).reshape(2, 1, 1))
            else:
                A_list.append(np.array([1, 0]).reshape(2, 1, 1))
        return cls(L, A_list, 0, hx, hz)

    @classmethod
    def load(cls, note):
        data = np.load(CACHE_DIR + note + '.npz')
        meta = data['meta']
        L = meta[0]
        ortho_center = meta[1]
        hx = meta[2]
        hz = meta[3]
        A_list = []
        for idx in range(int(L)):
            A_list.append(data[str(idx)])
        return cls(L, A_list, ortho_center, hx, hz)

    def save(self, note):
        meta = [self.L, self.ortho_center, self.hx, self.hz]
        data = {'meta': meta}
        for idx, A in enumerate(self.A_list):
            data[str(idx)] = A
        np.savez(CACHE_DIR + note + '.npz', **data)

    def norm(self):
        return self.dot(self)

    def dot(self, other):
        A = self.A_list[0]
        A_dag = np.transpose(other.A_list[0], axes=(0, 2, 1)).conj()
        contract = np.einsum('abc,acd->bd', A_dag, A, optimize=True)
        for j in range(1, self.L):
            A = self.A_list[j]
            A_dag = np.transpose(other.A_list[j], axes=(0, 2, 1)).conj()
            contract = np.einsum('abc,cd,ade->be', A_dag, contract, A, optimize=True)
        return np.trace(contract)

    def renormalize(self):
        # We renormalize at the orthogonality center so that we maintain the canonical form
        self.A_list[self.ortho_center] = self.A_list[self.ortho_center] / np.sqrt(self.norm())

    def get_state(self):
        output_state = []
        for state in range(2**self.L):
            prod = self.A_list[0][(state >> (self.L-1)) & 1].reshape(2)
            for j in range(1, self.L):
                prod = np.einsum('a,ab->b', prod, self.A_list[j][(state >> (self.L-1-j)) & 1], optimize=True)
            # Prod should only have one element
            output_state.append(prod[0])
        return np.array(output_state)

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
            W = np.einsum('abc,dce->adbe', self.A_list[j], self.A_list[j+1], optimize=True)
            # multiply by the 2-site operator
            W = np.einsum('ab,abcd->abcd', exp_xor, W, optimize=True)
            self.A_list[j], self.A_list[j+1] = self.__class__.__svd_W(W, left=False)

    def restore_canonical(self):
        self.ortho_center = 0
        self.move_ortho_center(self.L-1, k=None)

    def shift_ortho_center(self, k=None, left=True):
        if left:
            assert self.ortho_center != 0
            l_idx = self.ortho_center - 1
            r_idx = self.ortho_center
        else:
            assert self.ortho_center != self.L - 1
            l_idx = self.ortho_center
            r_idx = self.ortho_center + 1

        W = np.einsum('abc,dce->adbe', self.A_list[l_idx], self.A_list[r_idx], optimize=True)
        self.A_list[l_idx], self.A_list[r_idx] = self.__class__.__svd_W(W, k=k, left=left)

        if left:
            self.ortho_center -= 1
        else:
            self.ortho_center += 1

    def move_ortho_center(self, idx, k=None):
        distance = idx - self.ortho_center
        if distance != 0:
            left = distance < 0
            for _ in range(np.abs(distance)):
                self.shift_ortho_center(k=k, left=left)

    def imag_evolve(self, dt):
        self.__apply_field(dt)
        self.__apply_two_site(dt, 0)
        self.__apply_two_site(dt, 1)
        self.restore_canonical()

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
            traces = np.einsum('abc,dcb->ad', A_dag, A, optimize=True)
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

            A_prod = np.einsum('abc,acd->abd', A_dag, A, optimize=True)
            next_A_prod = np.einsum('abc,acd->abd', next_A, next_A_dag, optimize=True)

            for next_idx in range(2):
                for idx in range(2):
                    energy += self.xor[next_idx, idx] * np.einsum('ab,ba->', next_A_prod[next_idx], A_prod[idx], optimize=True)

            # Move the orthogonality center to the next site:
            self.shift_ortho_center(k=k, left=left)
        return energy

    def cool(self, dt, k):
        energy = [self.measure_energy(k=k)]
        while True:
            self.imag_evolve(dt)
            self.renormalize()
            energy.append(self.measure_energy(k=k))
            # We stop cooling when the slope of the energy density is below `TOL`.
            check = np.abs((energy[-1] - energy[-2]) / (self.L * dt))
            print(f'L={self.L} slope: {check}')
            if check < TOL:
                break
        energy = np.array(energy)
        return energy

    def correlation(self):
        assert self.ortho_center == 0

        sig_A_dag = np.transpose(self.A_list[0], axes=(0, 2, 1)).conj()
        # Applying the sigma_z operator:
        sig_A_dag[0] *= -1
        contract = np.einsum('abc,acd->abd', sig_A_dag, self.A_list[0], optimize=True)
        correl = [np.sum(np.diag(-1 * contract[0]) + np.diag(contract[1]))]
        contract = np.sum(contract, axis=0)
        for r in range(1, self.L):
            A_r = self.A_list[r]
            A_r_dag = np.transpose(A_r, axes=(0, 2, 1)).conj()

            contract = np.einsum('abc,cd,ade->abe', A_r_dag, contract, A_r, optimize=True)
            correl.append(np.sum(np.diag(-1 * contract[0]) + np.diag(contract[1])))
            contract = np.sum(contract, axis=0)
        return correl

    def real_evolve(self, dt):
        self.imag_evolve(1j * dt)

    def measure_sig_z(self):
        A = self.A_list[self.ortho_center]
        A_dag = np.transpose(A, axes=(0, 2, 1)).conj()
        # Apply sig_z operator by adding factor of -1
        A_dag[0] *= -1
        return np.einsum('abc,acb->', A_dag, A, optimize=True)

    def measure_sig_x(self):
        A = self.A_list[self.ortho_center]
        A_dag = np.transpose(A, axes=(0, 2, 1)).conj()
        # Apply sig_x operator by reversing `A_dag`
        A_dag = A_dag[::-1]
        return np.einsum('abc,acb->', A_dag, A, optimize=True)

    def get_schmidt_vals(self):
        assert self.ortho_center != self.L-1
        W = np.einsum('abc,dce->adbe', self.A_list[self.ortho_center], self.A_list[self.ortho_center+1], optimize=True)
        return self.__class__.__svd_W(W, return_s=True)

    def measure_entropy(self):
        probs = self.get_schmidt_vals()**2
        return -1 * np.sum(probs * np.log(probs))

    def time_trace(self, dt, k, N):
        data = {
            'sig_z_1': [],
            'sig_x_1': [],
            'sig_z_half': [],
            'sig_x_half': [],
            'entropy': []
        }
        self.move_ortho_center(0, k=None)
        data['sig_z_1'].append(self.measure_sig_z())
        data['sig_x_1'].append(self.measure_sig_x())
        self.move_ortho_center(self.L // 2, k=None)
        data['sig_z_half'].append(self.measure_sig_z())
        data['sig_x_half'].append(self.measure_sig_x())
        data['entropy'].append(self.measure_entropy())

        for i in range(N):
            print(f'Time trace step {i} of {N}')
            self.real_evolve(dt=dt)
            self.renormalize()

            self.move_ortho_center(self.L // 2, k=k)
            data['sig_z_half'].append(self.measure_sig_z())
            data['sig_x_half'].append(self.measure_sig_x())
            data['entropy'].append(self.measure_entropy())

            self.move_ortho_center(0, k=k)
            data['sig_z_1'].append(self.measure_sig_z())
            data['sig_x_1'].append(self.measure_sig_x())

        return data


@utility.cache('pkl', CACHE_DIR + 'real_time_trace')
def do_real_time_trace(mps, dt, k, N, note=None):
    return mps.time_trace(dt, k, N)


@utility.cache('pkl', CACHE_DIR + 'p4_3_2')
def cache_p4_3_2(L, N, dt=0.01, k=16, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz'], ising_hx=2, ising_hz=0, note=None):
    data = {}
    for ising in range(2):
        if ising:
            gnd_state = MPS.make_isotropic_state(L, np.arange(2), hx, hz)
        else:
            gnd_state = MPS.make_isotropic_state(L, np.arange(2), ising_hx, ising_hz)

        trial_eng = gnd_state.cool(dt, k)
        gnd_eng = trial_eng[-1]

        A = gnd_state.A_list[L // 2]
        new_A = {
            'x': np.array([A[1], A[0]]),
            'y': np.array([1j * A[1], -1j * A[0]]),
            'z': np.array([-1 * A[0], A[1]])
        }

        data[ising] = {}
        norm = gnd_state.norm()
        for dir in new_A:
            data[ising][dir] = [norm]

        for dir, op in new_A.items():
            A_list_copy = [A.copy() for A in gnd_state.A_list]
            mps = MPS(gnd_state.L, A_list_copy, gnd_state.ortho_center, gnd_state.hx, gnd_state.hz)
            mps.A_list[L // 2] = op.copy()

            A_list_copy = [A.copy() for A in gnd_state.A_list]
            mod_gnd_state = MPS(gnd_state.L, A_list_copy, gnd_state.ortho_center, gnd_state.hx, gnd_state.hz)
            mod_gnd_state.A_list[L // 2] = op.copy()
            for i in range(N):
                print(f'Time trace {ising} {dir} step {i} of {N}')
                mps.real_evolve(dt)
                mps.move_ortho_center(0, k=k)
                mps.renormalize()
                data[ising][dir].append(np.exp(1j * dt * (i+1) * gnd_eng) * mps.dot(mod_gnd_state))

    return data


def p4_1_fix_L(dtspace, L=12, k=16, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz']):
    eigs = dense_eigs(L, hx, hz, note=f'L{L}_hx{hx}_hz{hz}')
    gnd_eng = eigs['evals'][0]

    fig, axes = plt.subplots(figsize=(5, 5))
    for dt in dtspace:
        print(f'dt: {round(dt, 5)}')
        mps = MPS.make_isotropic_state(L, np.arange(2), hx, hz)
        energy = mps.cool(dt, k)

        tspace = dt * np.arange(0, len(energy))
        axes.plot(tspace, energy, label=rf'$\delta \tau = {round(dt, 5)}$')
        print(f'final energy: {energy[-1]}')
    print(f'true energy: {gnd_eng}')

    axes.axhline(gnd_eng, label='Ground State Energy', color='black', linestyle='dotted')
    axes.set_xlabel(r'Imaginary Time $\tau$')
    axes.set_ylabel('Trial Energy')

    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_gnd_eng_convergence.png', **utility.FIG_SAVE_OPTIONS)


def p4_1_fix_dt(Lspace, dt=0.01, k=16, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz']):
    correl_Lspace = Lspace[-5:]

    ferro_eng = []
    gnd_eng = []
    correl = []
    for L in Lspace:
        print(f'L: {L}')
        mps = MPS.make_isotropic_state(L, np.arange(2), hx, hz)
        ferro_eng.append(mps.cool(dt, k))
        gnd_eng.append(ferro_eng[-1][-1])
        if L in correl_Lspace:
            correl.append(mps.correlation())
    gnd_eng = np.array(gnd_eng)

    neel_L = Lspace[-1]
    print(f'Neel L: {neel_L}')
    mps = MPS.make_neel_state(neel_L, hx, hz)
    neel_eng = mps.cool(dt, k)
    neel_correl = mps.correlation()

    bulk_eng = []
    for i in range(len(Lspace) - 1):
        bulk_eng.append((gnd_eng[i + 1] - gnd_eng[i]) / (Lspace[i+1] - Lspace[i]))
    bulk_eng = np.array(bulk_eng)

    L_step = Lspace[1] - Lspace[0]
    fig, axes = plt.subplots(figsize=(5, 5))
    axes.plot(Lspace, gnd_eng / Lspace, label=r'$E_0(L)$')
    axes.plot(Lspace[:-1], bulk_eng, label=rf'$(E_0(L + {L_step}) - E_0(L)) / {L_step}$')
    axes.set_xlabel(r'$L$')
    axes.set_ylabel('Energy per Site')
    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_L_infty.png', **utility.FIG_SAVE_OPTIONS)

    fig, axes = plt.subplots(figsize=(5, 5))
    for idx, L in enumerate(correl_Lspace):
        axes.plot(np.arange(L), correl[idx], label=rf'$L={L}$')
    axes.set_xlabel(r'$r$')
    axes.set_ylabel(r'$\langle \sigma_1^z \sigma_{1+r}^z \rangle$')
    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_correl.png', **utility.FIG_SAVE_OPTIONS)

    fig, axes = plt.subplots(figsize=(5, 5))
    ferro_tspace = dt * np.arange(len(ferro_eng[-1]))
    neel_tspace = dt * np.arange(len(neel_eng))
    axes.plot(ferro_tspace, ferro_eng[-1], label=rf'Ferromagnetic Initial State, $L={neel_L}$')
    axes.plot(neel_tspace, neel_eng, label=rf'Néel Pattern Initial State, $L={neel_L}$')
    axes.set_xlabel(r'Imaginary Time $\tau$')
    axes.set_ylabel('Trial Energy')
    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_neel_eng_comp.png', **utility.FIG_SAVE_OPTIONS)

    fig, axes = plt.subplots(figsize=(5, 5))
    axes.plot(np.arange(neel_L), correl[-1], label=rf'Ferromagnetic Initial State, $L={neel_L}$')
    axes.plot(np.arange(neel_L), neel_correl, label=rf'Néel Pattern Initial State, $L={neel_L}$')
    axes.set_xlabel(r'$r$')
    axes.set_ylabel(r'$\langle \sigma_1^z \sigma_{1+r}^z \rangle$')
    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_1_neel_correl_comp.png', **utility.FIG_SAVE_OPTIONS)


def p4_2(L, kspace, N, dt=0.01, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz']):
    xi = (1/2) * np.array([-np.sqrt(3), 1])

    data = {}
    for k in kspace:
        print(f'k: {k}')
        mps = MPS.make_isotropic_state(L, xi, hx=hx, hz=hz)
        data[k] = do_real_time_trace(mps, dt, k, N, note=f'L{L}_dt{dt}_k{k}')

    fig, axes = plt.subplots(2, len(kspace), sharex=True, sharey=True, figsize=(5*len(kspace), 10))
    for idx, k in enumerate(kspace):
        tspace = dt * np.arange(len(data[k]['sig_z_1']))
        axes[0, idx].plot(tspace, data[k]['sig_z_1'], label=r'$j=1$')
        axes[0, idx].plot(tspace, data[k]['sig_z_half'], label=r'$j=L/2$')

        axes[1, idx].plot(tspace, data[k]['sig_x_1'], label=r'$j=1$')
        axes[1, idx].plot(tspace, data[k]['sig_x_half'], label=r'$j=L/2$')

        axes[1, idx].set_xlabel(r'Real Time $t$')
        axes[0, idx].set_title(rf'$k={k}$')

    axes[0, 0].set_ylabel(r'$\langle \sigma_j^z \rangle$')
    axes[1, 0].set_ylabel(r'$\langle \sigma_j^x \rangle$')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, **utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_2_sigma.png', **utility.FIG_SAVE_OPTIONS)

    fig, axes = plt.subplots(figsize=(5, 5))
    for idx, k, in enumerate(kspace):
        tspace = dt * np.arange(len(data[k]['entropy']))
        axes.plot(tspace, data[k]['entropy'], label=rf'$k={k}$')
    axes.set_xlabel(r'Real Time $t$')
    axes.set_ylabel(r'$S_{L/2}$')

    fig.legend(**utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_2_entropy.png', **utility.FIG_SAVE_OPTIONS)


def p4_3_2(L, N, dt=0.01, k=16, hx=FIELD_VALS['hx'], hz=FIELD_VALS['hz'], ising_hx=2, ising_hz=0):
    data = cache_p4_3_2(L, N, dt=dt, k=k, hx=hx, hz=hz, ising_hx=ising_hx, ising_hz=ising_hz, note=f'L{L}_N{N}_dt{dt}_k{k}')

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    for ising in data:
        for idx, dir in enumerate(data[ising]):
            tspace = dt * np.arange(len(data[ising][dir]))
            if ising:
                label = rf'$h_x = {ising_hx}, h_z = {ising_hz}$'
            else:
                label = rf'$h_x = {hx}, h_z = {hz}$'
            axes[idx].plot(tspace, data[ising][dir], label=label)
            axes[idx].set_title(rf'$\mu = {dir}$')
            axes[idx].set_xlabel(r'Real Time $t$')
    axes[0].set_ylabel(r'$C^{\mu\mu}_{L/2, L/2}(t)$')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, **utility.LEGEND_OPTIONS)
    fig.savefig(FIG_DIR + 'p4_3_2.png', **utility.FIG_SAVE_OPTIONS)


if __name__ == '__main__':
    DTSPACE = np.array([0.1, 0.01, 0.001])
    LSPACE = np.arange(10, 260, 10)
    KSPACE = np.array([8, 16, 32])
    TIME_STEPS = int(1e3)

    p4_1_fix_L(DTSPACE)

    p4_1_fix_dt(LSPACE)

    p4_2(LSPACE[-1], KSPACE, TIME_STEPS)

    p4_3_2(LSPACE[-1], TIME_STEPS)
