import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

FIG_DIR = './figs/'
CACHE_DIR = './data/'


class MPS():
    def __init__(self, L, A1, A, AL):
        """

        :param A1: A^1 tensor, organized in row vectors
        :param A: List of A^j tensors. Required that `A.shape[0] == L-2`.
        :param AL: A^L tensor, organized in column vectors.
        """
        self.L = L
        self.A1 = A1
        self.A = A
        self.AL = AL

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
        M = state.reshape(2, 2 ** (L - 1))
        U, s, Vh = np.linalg.svd(M, full_matrices=False)
        A1 = U[:, :k]
        W = np.einsum('a,ab->ab', s[:k], Vh[:k])
        AL = cls.make_from_state_helper(W, k, 2, L, A_list)
        return cls(L, A1, np.array(A_list), AL)

    @classmethod
    def make_from_state_helper(cls, arr, k, A_counter, L, A_list):
        """
        Helper method for `make_from_state`. Appends A^{`A_counter`} to the end of `A_list`.
        :param arr: Next W tensor to perform an SVD on.
        :param k: Bond dimension
        :param A_counter: See general function description above.
        :param L: System size.
        :param A_list: Growing list for the A^j tensors.
        :return: A^L.
        """
        M = arr.reshape(arr.shape[0] * 2, arr.shape[1] // 2)
        U, s, Vh = np.linalg.svd(M, full_matrices=False)

        # Even indices of `U` become (A^j)^0, and odd indices (A^j)^1
        A = np.stack((U[::2, :k], U[1::2, :k]))
        A_list.append(A)

        W = np.einsum('a,ab->ab', s[:k], Vh[:k])
        if A_counter == L - 1:
            return W
        else:
            return cls.make_from_state_helper(W, k, A_counter + 1, L, A_list)

    def apply_H(self, hx, hz, dt):
        H_A1 = self.A1.copy()
        H_A = self.A.copy()
        H_AL = self.AL.copy()

        # Applying the field operator first.
        mag = np.sqrt(hx**2 + hz**2)
        phi = dt * mag
        # Slightly modified matrix than that in Equation 20.
        # This is because we work in a basis were the zero index corresponds to spin down.
        n_dot_sigma = np.array([
            [hz, -hx],
            [-hx, -hz]
        ]) / mag
        field = np.cosh(phi) * np.identity(2) - np.sinh(phi) * n_dot_sigma
        H_A1 = np.einsum('ab,bc->ac', field, H_A1)
        H_A = np.einsum('ab,bcd->acd', field, H_A)
        H_AL = np.einsum('ab,cb->ac', field, H_A)
        # H_A1 = np.tensordot(field, H_A1, axes=([1], [0]))
        # H_A = np.tensordot(field, H_A, axes=([1], [0]))
        # H_AL = np.tensordot(field, H_AL, axes=([1], [1]))

        # Applying H_odd exponential second
        for j in range(0, self.L, 2):
            pass

        # Applying H_even exponential third


def p4_1(lspace):
    for L in lspace:
        state = np.zeros(2**L)
        state[-1] = 1
        mps = MPS.make_from_state(state, 1, L)
        print(mps.A1)
        print(mps.A)
        print(mps.AL)

        print(mps.A1.shape)
        print(mps.A.shape)
        print(mps.AL.shape)
        quit()



if __name__ == '__main__':
    lspace = np.arange(5, 10)

    p4_1(lspace)

