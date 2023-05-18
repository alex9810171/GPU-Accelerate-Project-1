import os, time
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import scipy.sparse.linalg as sla
from cupyx.scipy.sparse.linalg import lsqr, spilu, spsolve

DATA_TYPE = cp.float64

def get_matrix_a(size):
    matrix = cp.zeros((size, size), dtype=DATA_TYPE)
    for i in range(size):
            for j in range(size):
                if i==j:
                    matrix[i][j] = -4
                elif i+1==j:
                    matrix[i][j] = 1
                elif i==j+1:
                    matrix[i][j] = 1
                elif i+100==j:
                    matrix[i][j] = 1
                elif i==j+100:
                    matrix[i][j] = 1
    return matrix

def get_matrix_b(size):
    matrix = cp.zeros((size), dtype=DATA_TYPE)
    for i in range(size):
        matrix[i] = i+1
    return matrix

def matrix_inv(output_path, now):
    result_txt_path = os.path.join(output_path, now+'_result.txt')
    size = 105
    with open(result_txt_path, 'w') as f:
        # Get matrix a
        ## get value
        matrix_a = get_matrix_a(size=size)

        ## print value
        print(f'Matrix A (data_type={matrix_a.dtype}):', file=f)
        for i in range(size):
            print(*matrix_a[i], file=f)
        print('', file=f)
        
        ## compress value
        matrix_a = csr_matrix(matrix_a)

        # Get matrix b
        ## get value
        matrix_b = get_matrix_b(size=size)
        
        ## print value
        print(f'Matrix B (data_type={matrix_b.dtype}):', file=f)
        print(matrix_b, file=f)
        print('', file=f)
        
        # Start calculate matrix c
        ## get value
        start = time.time()
        matrix_c = lsqr(matrix_a, matrix_b)
        end = time.time()

        ## print value
        print(f'Matrix C (data_type={matrix_c[0].dtype}):', file=f)
        for i in range(matrix_c[0].shape[0]):
            print(matrix_c[0][i], end=' ', file=f)
            if (i+1)%10 == 0:
                print('', end='\n', file=f)
        
        # Print time.
        print(f'Time eclipse: {end-start}s')
        print(f'\nTime eclipse: {end-start}s', file=f)