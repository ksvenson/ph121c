import numpy
import numpy as np
import scipy

# 4.1
Lspace = (8, 10, 12, 14)
hspace = np.linspace(-1, 1, 1000)
for L in Lspace:
    dim = 2**L
    H = np.zeros((dim, dim), dtype=int)
    for j in range(dim-1):
        term = np.identity(dim, dtype=int)
