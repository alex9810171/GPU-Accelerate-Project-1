import os, time
from datetime import datetime
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spilu
import numba as nb
DATA_TYPE_1 = np.float64
DATA_TYPE_2 = np.int32
SAVE_PATH = 'spilu'
FILE_NAME = 'local2Dxy.txt'

x_domain_size = 100
y_domain_size = 300
dx = 0.5
dy = 0.25
D_x = 0.3                           # small particle Diffusivity for x-dir
D_y = 0.3                           # small particle Diffusivity for y-dir
k_small = 0.1                       # small overall mass transfer coeff
vmax = 1.0                            # vmax
rhobulk = 0.3                       # small bulk concentration
deltav = 15810
delta = 1e-6
m = int(x_domain_size/dx)
n = int(y_domain_size/dy)
size = (n+1)*(m+1)
Vx = np.zeros(size, dtype=DATA_TYPE_1)
y = np.zeros(size, dtype=DATA_TYPE_1)
now = datetime.now().strftime('%Y%m%d_%H%M%S')
result_path = os.path.join('test', SAVE_PATH)

rhosmall=np.zeros(size, dtype=DATA_TYPE_1)

@nb.jit(nopython=True)
def f0(f00):
    return f00 * np.pi / np.float64(6.0)
@nb.jit(nopython=True)
def f1(f11):
    return (np.float64(3.0) * f11**3 - np.float64(9.0) * f11**2 + \
            np.float64(8.0) * f11) / (np.float64(1.0) - f11)**3
@nb.jit(nopython=True)
def f2(f22):
    return (-np.float64(2.0) * f22**2 + np.float64(8.0) * f22) / (np.float64(1.0) - f22)**4
@nb.jit(nopython=True)
def f3(f33):
    return (np.float64(8.0) + np.float64(20.0) * f33 - np.float64(4.0) * f33**2) / (np.float64(1.0) - f33)**5
@nb.jit(nopython=True)
def f(rholeft, rhoright, rhomid, rhoup, rhodown, vx):
    return (
        vx / (2.0 * dx) * (rhoright - rholeft)
        - D_x* ((rhoright + rholeft - 2.0 * rhomid) / (dx ** 2.0) *\
         (1.0 + f2(f0(rhomid))) + (np.pi / 6.0) * f3(f0(rhomid)) * ((rhoright - rholeft) / (2.0 * dx)) ** 2.0)
        - D_y * ((rhoup + rhodown - 2.0 * rhomid) / (dy ** 2.0) *\
         (1.0 + f2(f0(rhomid))) + (np.pi / 6.0) * f3(f0(rhomid)) * ((rhoup - rhodown) / (2.0 * dy)) ** 2.0)
    )
@nb.jit(nopython=True)
def fb(rholeft, rhomid, rhoup, rhodown, vx):
    return (
        vx / (2.0 * dx) * (rhomid - rholeft)
        - D_x * ((rholeft - rhomid) / (dx ** 2.0) *\
         (1.0 + f2(f0(rhomid))) + (np.pi / 6.0) * f3(f0(rhomid)) * ((rhomid - rholeft) / (2.0 * dx)) ** 2.0)
        - D_y * ((rhoup + rhodown - 2.0 * rhomid) / (dy ** 2.0) * \
        (1.0 + f2(f0(rhomid))) + (np.pi / 6.0) * f3(f0(rhomid)) * ((rhoup - rhodown) / (2.0 * dy)) ** 2.0)
    )
@nb.jit(nopython=True)
def f_jac_rholeft(rholeft, rhoright, rhomid, rhoup, rhodown, delta, vx):
    return (f(rholeft + delta, rhoright, rhomid, rhoup, rhodown, vx) - \
            f(rholeft, rhoright, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def f_jac_rhoright(rholeft, rhoright, rhomid, rhoup, rhodown, delta, vx):
    return (f(rholeft, rhoright + delta, rhomid, rhoup, rhodown, vx) - \
            f(rholeft, rhoright, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def f_jac_rhomid(rholeft, rhoright, rhomid, rhoup, rhodown, delta, vx):
    return (f(rholeft, rhoright, rhomid + delta, rhoup, rhodown, vx) - \
            f(rholeft, rhoright, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def f_jac_rhoup(rholeft, rhoright, rhomid, rhoup, rhodown, delta, vx):
    return (f(rholeft, rhoright, rhomid, rhoup + delta, rhodown, vx) - \
            f(rholeft, rhoright, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def f_jac_rhodown(rholeft, rhoright, rhomid, rhoup, rhodown, delta, vx):
    return (f(rholeft, rhoright, rhomid, rhoup, rhodown + delta, vx) - \
            f(rholeft, rhoright, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def fb_jac_rholeft(rholeft, rhomid, rhoup, rhodown, delta, vx):
    return (fb(rholeft + delta, rhomid, rhoup, rhodown, vx) - \
            fb(rholeft, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def fb_jac_rhomid(rholeft, rhomid, rhoup, rhodown, delta, vx):
    return (fb(rholeft, rhomid + delta, rhoup, rhodown, vx) - \
            fb(rholeft, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def fb_jac_rhoup(rholeft, rhomid, rhoup, rhodown, delta, vx):
    return (fb(rholeft, rhomid, rhoup + delta, rhodown, vx) - \
            fb(rholeft, rhomid, rhoup, rhodown, vx)) / delta
@nb.jit(nopython=True)
def fb_jac_rhodown(rholeft, rhomid, rhoup, rhodown, delta, vx):
    return (fb(rholeft, rhomid, rhoup, rhodown +delta, vx) - \
            fb(rholeft, rhomid, rhoup, rhodown, vx)) / delta  

def get_Vx_y():
    indices = np.arange(size)
    y = indices%(n+1)    
    Vx = vmax*(1-(y-deltav)**2/deltav**2)
    return Vx
        #Vx[i] = 0.0


def get_RHS():
    RHS=np.zeros(size, dtype=DATA_TYPE_1)
    indices=np.arange(size)
    condition1 = ((indices+1)<=(n+1))
    condition2 = np.logical_and((indices+1)>(n+1) , (indices+1)<=m*(n+1))
    condition3 = ((indices+1)>m*(n+1))
    condition4 = np.logical_and((indices+1)%(n+1)!=0 , (indices+1)%(n+1)!=1)
    condition5 = ((indices+1)%(n+1)==0)
    condition6 = ((indices+1)%(n+1)==1)
    mask1 = condition1
    mask2 = condition2 & condition4
    mask3 = condition2 & condition5
    mask4 = condition2 & condition6
    mask5 = condition3 & condition4
    mask6 = condition3 & condition5
    mask7 = condition3 & condition6

    RHS[mask1] = rhosmall[indices[mask1]]-rhobulk
    RHS[mask2] = f(rhosmall[indices[mask2]-(n+1)],(rhosmall[indices[mask2]+(n+1)]),(rhosmall[indices[mask2]])\
				,(rhosmall[indices[mask2]+1]),(rhosmall[indices[mask2]-1]),Vx[indices[mask2]])
    RHS[mask3] = rhosmall[indices[mask3]]-rhobulk  
    RHS[mask4] = rhosmall[indices[mask4]]-0.0
    RHS[mask5] = fb(rhosmall[indices[mask5]-(n+1)],(rhosmall[indices[mask5]])\
				,(rhosmall[indices[mask5]+1]),(rhosmall[indices[mask5]-1]),Vx[indices[mask5]])
    RHS[mask6] = rhosmall[indices[mask6]]-rhobulk  
    RHS[mask7] = rhosmall[indices[mask7]]-0.0
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
        
    def get_upper(self):  #向量化操作

        self.upper_col_indx=np.arange(self.size-1)
        self.upper_row_ptr[:1]=0
        self.upper_row_ptr[1:]=np.arange(self.size) 
        indices=np.arange(self.size-1)
        condition1 = np.logical_and((indices+1) > (n + 1), (indices+1) <= m * (n + 1))
        condition2 = np.logical_and((indices+1) % (n + 1) != 0, (indices+1) % (n + 1) != 1)
        condition3 = (indices+1) > m * (n + 1)
        mask1=condition1&condition2
        mask2=condition3&condition2

        self.upper_values[mask1] = f_jac_rhoup(rhosmall[indices[mask1]-(n+1)],(rhosmall[indices[mask1]+(n+1)]),(rhosmall[indices[mask1]])\
				,(rhosmall[indices[mask1]+1]),(rhosmall[indices[mask1]-1]),delta,Vx[indices[mask1]])
        self.upper_values[mask2] = fb_jac_rhoup(rhosmall[indices[mask2]-(n+1)],(rhosmall[indices[mask2]])\
				,(rhosmall[indices[mask2]+1]),(rhosmall[indices[mask2]-1]),delta,Vx[indices[mask2]])
              
        return csc_matrix((self.upper_values, self.upper_col_indx, 
                            self.upper_row_ptr), shape=(self.size, self.size))
        
    def get_lower(self):

        self.lower_col_indx = np.arange(self.size-1)+1
        self.lower_row_ptr[:size] = np.arange(self.size)
        self.lower_row_ptr[size:] = self.size-1
        indices=np.arange(self.size-1)
        condition1 = np.logical_and((indices+2) > (n + 1), (indices+2) <= m * (n + 1))
        condition2 = np.logical_and((indices+2) % (n + 1) != 0, (indices+2) % (n + 1) != 1)
        condition3 = (indices+2) > m * (n + 1)
        mask1=condition1&condition2
        mask2=condition3&condition2     

        self.lower_values[mask1] = f_jac_rhodown(rhosmall[indices[mask1]-(n+1)+1],(rhosmall[indices[mask1]+(n+1)+1]),(rhosmall[indices[mask1]+1])\
				,(rhosmall[indices[mask1]+1+1]),(rhosmall[indices[mask1]-1+1]),delta,Vx[indices[mask1]+1])
        self.lower_values[mask2] = fb_jac_rhodown(rhosmall[indices[mask2]-(n+1)+1],(rhosmall[indices[mask2]+1])\
				,(rhosmall[indices[mask2]+1+1]),(rhosmall[indices[mask2]-1+1]),delta,Vx[indices[mask2]+1])       
        return csc_matrix((self.lower_values, self.lower_col_indx, 
                            self.lower_row_ptr), shape=(self.size, self.size))
        
    def get_medium(self):

        self.medium_col_indx = np.arange(self.size)
        self.medium_row_ptr = np.arange(self.size+1)
        indices=np.arange(self.size)
        condition1 = np.logical_and((indices+1) > (n + 1), (indices+1) <= m * (n + 1))
        condition2 = np.logical_and((indices+1) % (n + 1) != 0, (indices+1) % (n + 1) != 1)
        condition3 = (indices+1) > m * (n + 1)
        mask1=condition1&condition2
        mask2=condition3&condition2

        self.medium_values[mask1] = f_jac_rhomid(rhosmall[indices[mask1]-(n+1)],(rhosmall[indices[mask1]+(n+1)]),(rhosmall[indices[mask1]])\
				,(rhosmall[indices[mask1]+1]),(rhosmall[indices[mask1]-1]),delta,Vx[indices[mask1]])
        self.medium_values[mask2] = fb_jac_rhomid(rhosmall[indices[mask2]-(n+1)],(rhosmall[indices[mask2]])\
				,(rhosmall[indices[mask2]+1]),(rhosmall[indices[mask2]-1]),delta,Vx[indices[mask2]]) 
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
        
        self.upper_col_indx = np.arange(size-(n+1))        
        self.upper_row_ptr[:(n+1)] = 0
        self.upper_row_ptr[(n+1):] = np.arange(size-(n+1)+1)
        indices=np.arange(self.size-(n+1))
        condition1 = (indices) >  (n + 1)
        condition2 = np.logical_and((indices+1) % (n + 1) != 0, (indices+1) % (n + 1) != 1)
        mask1=condition1&condition2

        self.upper_values[mask1] = f_jac_rhoright(rhosmall[indices[mask1]-(n+1)],(rhosmall[indices[mask1]+(n+1)]),(rhosmall[indices[mask1]])\
				,(rhosmall[indices[mask1]+1]),(rhosmall[indices[mask1]-1]),delta,Vx[indices[mask1]])
        return csc_matrix((self.upper_values, self.upper_col_indx, 
                            self.upper_row_ptr), shape=(self.size, self.size))
    
    def get_lower(self):

        self.lower_col_indx = np.arange(self.size-(n+1))+(n+1)
        self.lower_row_ptr[:(self.size-(n+1))] = np.arange(self.size-(n+1))
        self.lower_row_ptr[(self.size-(n+1)):] = (self.size-(n+1))
        indices=np.arange(self.size-(n+1))
        condition1 = ((indices+1) <= (m-1) * (n+1))
        condition2 = np.logical_and((indices+1) % (n+1) != 0, (indices+1) % (n+1) != 1)
        condition3 = ((indices+1) > (m-1) * (n+1))
        mask1=condition1&condition2
        mask2=condition3&condition2

        self.lower_values[mask1] = f_jac_rholeft(rhosmall[indices[mask1]-(n+1)+(n+1)],(rhosmall[indices[mask1]+(n+1)+(n+1)]),(rhosmall[indices[mask1]+(n+1)])\
				,(rhosmall[indices[mask1]+1+(n+1)]),(rhosmall[indices[mask1]-1+(n+1)]),delta,Vx[indices[mask1]+(n+1)])

        self.lower_values[mask2] = fb_jac_rholeft(rhosmall[indices[mask2]-(n+1)+(n+1)],(rhosmall[indices[mask2]+(n+1)])\
				,(rhosmall[indices[mask2]+1+(n+1)]),(rhosmall[indices[mask2]-1+(n+1)]),delta,Vx[indices[mask2]+(n+1)])
        return csc_matrix((self.lower_values, self.lower_col_indx, 
                            self.lower_row_ptr), shape=(self.size, self.size))

def compute_tridiagonal(output_path, now):
    # construct matrix
    start = time.time()
    total_start = time.time()  
    get_Vx_y()
    error=1.0
    iter=1
    while error >= 1e-6 :
        print(f'iter: {iter}')       
        f = F(size)
        t = T(size)
        matrix = (f.get_lower()+f.get_upper()+t.get_lower()+t.get_medium()+t.get_upper())
        end = time.time()
        #print(matrix)
        if iter%5==0:
            print(f'INFO     | local2Dxy_scipy - matrix construction time: {end-start} s')
       
        # construct RHS
        start = time.time()
        RHS = get_RHS()
        end = time.time()
        if iter%5==0:
            print(f'INFO     | local2Dxy_scipy - RHS construction time: {end-start} s')

        # compute LU
        start = time.time()
        lu=spilu(matrix, drop_tol=0, fill_factor=int(((n+1)*(m+1))**0.5)/5)
        end = time.time()
        if iter%5==0:
            print(f'INFO     | local2Dxy_scipy - CPU incomplete LU decomposition time: {end-start} s')

        start = time.time()
        displacement=lu.solve(RHS)
        global rhosmall 
        rhosmall = rhosmall-displacement
        end = time.time()
        if iter%5==0:
            print(f'INFO     | local2Dxy_scipy - CPU solving linear equations time: {end-start} s')
        error=max(abs(displacement))
        total_end = time.time() 
        print(f'INFO     | local2Dxy_scipy - CPU total time: {total_end-total_start} s, iter_error: {error}')
        iter=iter+1
        # write file
  
def write(output_path, now):
    start=time.time()
    result_txt_path = os.path.join(output_path, FILE_NAME)
    array_2d = rhosmall.reshape((m+1, n+1))
    array_2d_new=np.flipud(np.rot90(array_2d))
    with open(result_txt_path, 'w') as file:
        for row in array_2d_new:
            # 将每个元素转换为字符串，指定小数点后 15 位的精度
            row_str = [f"{num:.15f}" for num in row]
            # 将每个元素以制表符分隔并写入文件
            file.write('\t'.join(row_str))
            file.write('\n')
    end = time.time()
    print(f'INFO     | local2Dxy_scipy - write file time: {end-start} s')
def main():
    os.makedirs(result_path, exist_ok=True)
    compute_tridiagonal(result_path, now)
    write(result_path, now)
if __name__ == '__main__':
    main()