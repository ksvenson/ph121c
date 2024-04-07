import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# 4.1
Lspace = (8, 10, 12, 14)
hspace = np.linspace(-1, 1, 10)

plt.figure()

for L in Lspace:
    dim = 2**L
    H = np.zeros((len(hspace), dim, dim))
    for i in range(dim):
        H[:, i, i] += 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)
        for flip in range(0, L):
            H[:, i ^ (1 << flip), i] -= hspace
    gnd_eng = []
    for i in range(len(hspace)):
        print(f'finding evals: h={hspace[i]}, L={L}')
        evals = sp.linalg.eigvals(H[i])
        gnd_eng.append(np.min(evals))
    plt.plot(hspace, gnd_eng, label=rf'$L={L}$')
plt.show()
