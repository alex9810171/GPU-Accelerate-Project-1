import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
A = csc_matrix([[1., 0.], [1., 1.], [0., 1.]], dtype=float)
b = np.array([0., 0., 0.], dtype=float)
x, istop, itn, normr = lsqr(A, b)