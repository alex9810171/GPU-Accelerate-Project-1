import os, sys, time
import numpy as np
import scipy
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import scipy.sparse.linalg as sla
from cupyx.scipy.sparse.linalg import lsqr, spilu, spsolve
#from src import utils

def main():
    '''
    # Load parameters & time.
    param, now = utils.get_config_and_time()

    # Create new result file directory.
    result_path = os.path.join(param['result_files']['path'], now)
    print(f'Create train file directory in {result_path}')
    os.makedirs(result_path, exist_ok=True)

    # Check gpu.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to compute...')
    '''
    path = 'output.txt'
    f = open(path, 'w')
    # Compute.
    size = 10
    
    A = [[0.0]*size for i in range(size)]
    for i in range(size):
        for j in range(size):
            if i==j:
                A[i][j] = -4.0
            elif i+1==j:
                A[i][j] = 1.0
            elif i==j+1:
                A[i][j] = 1.0
            elif i+100==j:
                A[i][j] = 1.0
            elif i==j+100:
                A[i][j] = 1.0
    
    A = cp.array(A)
    #for i in range(size):
    #    print(*A[i], file=f)
    #print('', file=f)
    
    A = csr_matrix(A)

    #B = cp.random.rand(size)
    B = cp.zeros((size), dtype=cp.float32)
    for i in range(size):
        B[i] = i+1
    print(B)
    #print(B, file=f)
    start = time.time()
    C = spsolve(A, B)
    print(C)
    end = time.time()
    
    print(f'Time eclipse: {end-start}s')
    #print('',file=f)
    #print(C[0], end=' ', file=f)
    #for i in range(size):
    #    for j in range(size):
    #        print(C[0][i, j], end=' ', file=f)
    #    print('\n',file=f)
    for i in range(len(C)):
        print(C[i], end=' ', file=f)
        if (i+1)%10 == 0:
            print('', end='\n', file=f)
    f.close()

    #B = torch.randn(15000, 1)
    #start = time.time()
    #X = torch.linalg.solve(A,B)
    #end = time.time()
    #print(A)
    #print(f'Time eclipse: {end-start}s')


if __name__ == '__main__':
    main()