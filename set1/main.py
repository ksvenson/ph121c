import numpy
import numpy as np
import scipy

# 4.1
Lspace = (3, 8, 10, 12, 14)
hspace = np.linspace(-1, 1, 1000)
for L in Lspace:
    dim = 2**L
    H = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        print('='*50)
        print(bin(i))
        print(bin(i & ~(1 << L-1)))
        print(bin(i >> 1))
        H[i, i] = 2 * ((i & ~(1 << L-1)) ^ (i >> 1)).bit_count() - (L-1)
        print(H[i, i])
    print(H)
    break
