import os, time
from datetime import datetime
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu

DATA_TYPE_1 = np.float64
DATA_TYPE_2 = np.int32
SAVE_PATH = 'spilu'
FILE_NAME = 'ideal2Dxy.txt'

x_domain_size = 770
y_domain_size = 770
dx = 1.0
dy = 1.0
D_x = 0.3                           # small particle Diffusivity for x-dir
D_y = 0.3                           # small particle Diffusivity for y-dir
k_small = 0.1                       # small overall mass transfer coeff
vmax = 1                            # vmax
rhobulk = 0.3                       # small bulk concentration
deltav = 15810
m = int(x_domain_size/dx)
n = int(y_domain_size/dy)
size = (n+1)*(m+1)
Vx = np.zeros(size, dtype=DATA_TYPE_1)
y = np.zeros(size, dtype=DATA_TYPE_1)
now = datetime.now().strftime('%Y%m%d_%H%M%S')
result_path = os.path.join('test', SAVE_PATH)

def get_Vx_y():
    for i in range(size):
        y[i] = dy * float(i)
        while y[i] > y_domain_size:
            y[i] = y[i-(n+1)]    
        Vx[i] = vmax*(1-(y[i]-deltav)**2/deltav**2)

def get_RHS():
    RHS=np.zeros(size, dtype=DATA_TYPE_1)
    for i in range(size):
        if (i+1)<=(n+1):
            RHS[i] = rhobulk
        else:
            if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
                RHS[i] = 0
            elif (i+1)%(n+1)==0:
                RHS[i] = rhobulk  
            else: 
                RHS[i] = 0
    return RHS

class T():
    def __init__(self, size):
        self.size = size
        self.medium_values = np.ones(size, dtype=DATA_TYPE_1)
        self.medium_col_indx = np.zeros(size, dtype=DATA_TYPE_2)
        self.medium_row_ptr = np.zeros(size+1, dtype=DATA_TYPE_2)
        
        self.upper_values = np.zeros(size-1, dtype=DATA_TYPE_1)
        self.upper_col_indx = np.zeros(size-1, dtype=DATA_TYPE_2)
        self.upper_row_ptr = np.zeros(size+1, dtype=DATA_TYPE_2)
        
        self.lower_values = np.zeros(size-1, dtype=DATA_TYPE_1)
        self.lower_col_indx = np.zeros(size-1, dtype=DATA_TYPE_2)
        self.lower_row_ptr = np.zeros(size+1, dtype=DATA_TYPE_2)
    
    def get_upper(self):
        for i in range(self.size-1):
            self.upper_col_indx[i] = i
        for i in range(self.size+1):
            if i<=1:
                self.upper_row_ptr[i] = 0
            else:
                self.upper_row_ptr[i] = i-1
        for i in range(self.size-1):
            if (i+1)>(n+1):
                if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
                    self.upper_values[i] = -D_y/(dy**2)
        return csc_matrix((self.upper_values, self.upper_col_indx, 
                            self.upper_row_ptr), shape=(self.size, self.size))
        
    def get_lower(self):
        for i in range(self.size-1):
            self.lower_col_indx[i] = i+1
        for i in range(self.size+1):
            if i==self.size:
                self.lower_row_ptr[i] = self.size-1
            else:
                self.lower_row_ptr[i] = i
        for i in range(self.size-1):
            if (i+2)>(n+1):
                if (i+2)%(n+1)!=0 and (i+2)%(n+1)!=1:
                    self.lower_values[i] = -D_y/(dy**2)
        return csc_matrix((self.lower_values, self.lower_col_indx, 
                            self.lower_row_ptr), shape=(self.size, self.size))
        
    def get_medium(self):
        for i in range(self.size):
            self.medium_col_indx[i] = i
        for i in range(self.size+1):
            self.medium_row_ptr[i] = i
        for i in range(self.size):
            if (i+1)>(n+1) and (i+1)<=(n+1)*m:
                if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
                    self.medium_values[i] = 2*D_x/(dx**2)+2*D_y/(dy**2)
            elif (i+1) > (n+1)*m:
                if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
                    self.medium_values[i] = D_x/(dx**2)+2*D_y/(dy**2)+Vx[i]/(2**dx)
        return csc_matrix((self.medium_values, self.medium_col_indx, 
                            self.medium_row_ptr), shape=(self.size, self.size))

class F():
    def __init__(self, size):
        self.size = size
        
        self.upper_values = np.zeros(size-(n+1), dtype=DATA_TYPE_1)
        self.upper_col_indx = np.zeros(size-(n+1), dtype=DATA_TYPE_2)
        self.upper_row_ptr = np.zeros(size+1, dtype=DATA_TYPE_2)
        
        self.lower_values = np.zeros(size-(n+1), dtype=DATA_TYPE_1)
        self.lower_col_indx = np.zeros(size-(n+1), dtype=DATA_TYPE_2)
        self.lower_row_ptr = np.zeros(size+1, dtype=DATA_TYPE_2)
    
    def get_upper(self):
        for i in range(self.size-(n+1)):
            self.upper_col_indx[i] = i
        for i in range(self.size+1):
            if i<=(n+1):
                self.upper_row_ptr[i] = 0
            else:
                self.upper_row_ptr[i] = i-(n+1)
        for i in range(self.size-(n+1)):
            if i>(n+1):
                if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
                    self.upper_values[i] = -D_x/(dx**2)+Vx[i]/(2**dx)
        return csc_matrix((self.upper_values, self.upper_col_indx, 
                            self.upper_row_ptr), shape=(self.size, self.size))
    
    def get_lower(self):
        for i in range(self.size-(n+1)):
            self.lower_col_indx[i] = i+(n+1)
        for i in range(self.size+1):
            if i>=size-(n+1):
                self.lower_row_ptr[i] = size-(n+1)
            else:
                self.lower_row_ptr[i] = i
        for i in range(self.size-(n+1)):
            if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
                self.lower_values[i] = -D_x/dx**2-Vx[i]/(2**dx)
        return csc_matrix((self.lower_values, self.lower_col_indx, 
                            self.lower_row_ptr), shape=(self.size, self.size))

def compute_tridiagonal(output_path, now):
    # construct matrix
    start = time.time()
    f = F(size)
    t = T(size)
    matrix = (f.get_lower()+f.get_upper()+t.get_lower()+t.get_medium()+t.get_upper())
    end = time.time()
    print(f'matrix construction time: {end-start} s')
    
    # construct RHS
    RHS = get_RHS()

    # compute LU
    start=time.time()
    lu=spilu(matrix, drop_tol=0, fill_factor=40)
    rho=lu.solve(RHS)
    end = time.time()
    print(f'matrix solving time: {end-start} s')
    
    # write file
    result_txt_path = os.path.join(output_path, FILE_NAME)
    with open(result_txt_path, 'w') as file:
        for i in range(size):
            if (i+1)%(n+1) != 0:
                file.write(f'{rho[i]:.15f} ')
            else:
                file.write(f'{rho[i]:.15f} \n')

def main():
    os.makedirs(result_path, exist_ok=True)
    get_Vx_y()
    compute_tridiagonal(result_path, now)

if __name__ == '__main__':
    main()