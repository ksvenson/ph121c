import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

FIG_DIR = './figs/'
CACHE_DIR = './data/'

FIELD_VALS = {'hx': -1.05, 'hz': 0.5}


class MPS:
    """
    Represents an MPS state.
    """
    def __init__(self, L, A_list, ortho_center):
        """
        Creates an MPS given the list of A tensors.
        :param L: System size.
        :param A_list: List of A^j tensors.
        """
        self.L = L
        self.A_list = A_list
        self.ortho_center = ortho_center

    @staticmethod
    def __svd_W(W, k=None, left=True):
        W = np.moveaxis(W, (0, 1, 2, 3), (1, 2, 0, 3))
        W.reshape(W.shape[0] * W.shape[1], W.shape[2] * W.shape[3])

        U, s, Vh = np.linalg.svd(W, full_matrices=False)
        if k is not None:
            U = U[:, :k]
            s = s[:k]
            Vh = Vh[:k]
        if left:
            Vh = np.einsum('a,ab->ab', s, Vh)
        else:
            U = np.einsum('ab,b->ab', U, s)
        split_idx = Vh.shape[1] // 2
        return np.stack((U[::2], U[1::2])), np.stack((Vh[:, :split_idx], Vh[:, split_idx:]))

    @classmethod
    def make_from_state(cls, state, k, L):
        """
        Returns an MPS representing `state` in the sigma z basis.
        :param state: State in the sigma z basis.
        :param k: Bond dimension.
        :param L: System size.
        :return: MPS object representing `state`.
        """
        A_list = []
        W = state.reshape(1, 2**L)
        cls.__make_from_state_helper(W, k, 1, L, A_list)
        return cls(L, A_list, L-1)

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

        W = np.einsum('a,ab->ab', s[:k], Vh[:k])
        if A_counter == L - 1:
            A_list.append(W[:k].T.reshape(2, 1, min(2, k)))
        else:
            cls.__make_from_state_helper(W, k, A_counter + 1, L, A_list)

    def time_evolve(self, hx, hz, dt, k):
        self.__apply_field(hx, hz, dt)
        self.__apply_two_site(dt, 0)
        self.__apply_two_site(dt, 1)
        self.__restore_canonical(k)

    def __apply_field(self, hx, hz, dt):
        mag = np.sqrt(hx ** 2 + hz ** 2)
        phi = dt * mag
        # Slightly modified matrix from Equation 20.
        # This is because we work in a basis were the zero index corresponds to spin down.
        n_dot_sigma = np.array([
            [hz, -hx],
            [-hx, -hz]
        ]) / mag
        field = np.cosh(phi) * np.identity(2) - np.sinh(phi) * n_dot_sigma
        for idx, A in enumerate(self.A_list):
            self.A_list[idx] = np.einsum('ab,bcd->acd', field, A)
        # H_A = np.tensordot(field, H_A, axes=([1], [0]))

    def __apply_two_site(self, dt, parity):
        # helper matrix
        xor = np.array([
            [1, -1],
            [-1, 1]
        ])
        xor = np.exp(dt * xor)
        for j in range(parity, self.L-1, 2):
            # multiply A^j and A^{j+1} together
            W = np.einsum('abc,dce->adbe', self.A_list[j], self.A_list[j+1])
            # multiply by the 2-site operator
            W = np.einsum('ab,abcd->abcd', xor, W)
            self.A_list[j], self.A_list[j+1] = self.__class__.__svd_W(W)

    def __restore_canonical(self, k):
        # First, we sweep left to right without truncating
        for j in range(self.L-1):
            W = np.einsum('abc,dce->adbe', self.A_list[j], self.A_list[j+1])
            self.A_list[j], self.A_list[j+1] = self.__class__.__svd_W(W)

        # Second, we sweep right to left with truncating
        for j in range(self.L-1, 0, -1):
            W = np.einsum('abc,dbe->adbe', self.A_list[j-1], self.A_list[j])
            self.A_list[j-1], self.A_list[j+1] = self.__class__.__svd_W(W, k=k, left=False)
        self.ortho_center = 0

    def measure_energy(self):



def p4_1(lspace, dt, N, hx, hz, k):
    for L in lspace:
        state = np.zeros(2**L)
        state[-1] = 1
        mps = MPS.make_from_state(state, 1, L)
        for _ in range(N):
            mps.time_evolve(hx, hz, dt, k)


if __name__ == '__main__':
    p4_1(np.arange(8, 9))

