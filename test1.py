import os, time
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import lsqr

indptr = cp.array([0, 2, 3, 6],dtype=cp.float64)
indices = cp.array([0, 2, 2, 0, 1, 2],dtype=cp.float64)
data = cp.array([1, 2, 3, 4, 5, 6],dtype=cp.float64)
a=csr_matrix((data, indices, indptr), shape=(3, 3))

indptr = cp.array([0, 0, 1, 1],dtype=cp.float64)
indices = cp.array([1],dtype=cp.float64)
data = cp.array([3],dtype=cp.float64)
b=csr_matrix((data, indices, indptr), shape=(3, 3))

print(a+b)