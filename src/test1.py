import os, time
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spilu
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import spsolve
from cupyx.profiler import benchmark
x_domain_size = 800.0
y_domain_size = 800.0
dx = 1.0
dy = 1.0
D_x= 0.3                # small particle Diffusivity for x-dir
D_y= 0.3                # small particle Diffusivity for y-dir
k_small= 0.1            # small overall mass transfer coeff
vmax= 1.0               # vmax
rhobulk= 0.3           # small bulk concentration
deltav= 15810
Data_type=np.float64
m  = int(x_domain_size/dx)
n  = int(y_domain_size/dy)
size=((n+1)*(m+1))
start1 = time.time()
Vx=np.zeros(size,dtype=Data_type)
y=np.zeros(size,dtype=Data_type)
for i in range(size):
    y[i] = dy * float(i)
    while y[i] > y_domain_size:
        y[i] = y[i-(n+1)]    
    Vx[i] = vmax * (1.0 - (y[i]-deltav)**2/deltav**2)



T_medium_col_indx=np.zeros(size,dtype=Data_type)
T_medium_row_ptr=np.zeros(size+1,dtype=Data_type)
T_medium_values=np.ones(size,dtype=Data_type)
T_upper_col_indx=np.zeros(size-1,dtype=Data_type)
T_upper_row_ptr=np.zeros(size+1,dtype=Data_type)
T_upper_values=np.zeros(size-1,dtype=Data_type)
T_lower_col_indx=np.zeros(size-1,dtype=Data_type)
T_lower_row_ptr=np.zeros(size+1,dtype=Data_type)
T_lower_values=np.zeros(size-1,dtype=Data_type)
F_upper_col_indx=np.zeros(size-(n+1),dtype=Data_type)
F_upper_row_ptr=np.zeros(size+1,dtype=Data_type)
F_upper_values=np.zeros(size-(n+1),dtype=Data_type)
F_lower_col_indx=np.zeros(size-(n+1),dtype=Data_type)
F_lower_row_ptr=np.zeros(size+1,dtype=Data_type)
F_lower_values=np.zeros(size-(n+1),dtype=Data_type)
RHS_col_indx=np.zeros(size,dtype=Data_type)
RHS_row_ptr=np.zeros(size+1,dtype=Data_type)
RHS_values=np.ones(size,dtype=Data_type)

for i in range(size):
    T_medium_col_indx[i]=i
for i in range(size+1):
    T_medium_row_ptr[i]=i
for i in range(size):
    if (i+1)>(n+1) and (i+1)<=(n+1)*m:
        if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
            T_medium_values[i]=2.0*D_x/(dx**2.0)+2.0*D_y/(dy**2.0)
    elif (i+1)>(n+1)*m:
        if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
            T_medium_values[i]=D_x/(dx**2.0)+2.0*D_y/(dy**2.0)+Vx[i]/(2.0**dx)
T_medium=csr_matrix((T_medium_values, T_medium_col_indx, 
                     T_medium_row_ptr), shape=(size, size))

for i in range(size-1):
    T_upper_col_indx[i]=i+1
for i in range(size+1):
    if i==size:
        T_upper_row_ptr[i]=size-1
    else:
        T_upper_row_ptr[i]=i
for i in range(size-1):
    if (i+1)>(n+1):
        if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
            T_upper_values[i]=-D_y/(dy**2.0)
 
T_upper=csr_matrix((T_upper_values, T_upper_col_indx, 
                     T_upper_row_ptr), shape=(size, size))


for i in range(size-1):
    T_lower_col_indx[i]=i
for i in range(size+1):
    if i<=1:
        T_lower_row_ptr[i]=0
    else:
        T_lower_row_ptr[i]=i-1
for i in range(size-1):
    if (i+2)>(n+1):
        if (i+2)%(n+1)!=0 and (i+2)%(n+1)!=1:
            T_lower_values[i]=-D_y/(dy**2.0)
 
T_lower=csr_matrix((T_lower_values, T_lower_col_indx, 
                     T_lower_row_ptr), shape=(size, size))

for i in range(size-(n+1)):
    F_upper_col_indx[i]=i+(n+1)
for i in range(size+1):
    if i>=size-(n+1):
        F_upper_row_ptr[i]=size-(n+1)
    else:
        F_upper_row_ptr[i]=i
for i in range(size-(n+1)):
    if (i+1)>(n+1):
        if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
            F_upper_values[i]=-D_x/dx**2+Vx[i]/(2.0**dx)
 
F_upper=csr_matrix((F_upper_values, F_upper_col_indx, 
                     F_upper_row_ptr), shape=(size, size))

for i in range(size-(n+1)):
    F_lower_col_indx[i]=i
for i in range(size+1):
    if i<=(n+1):
        F_lower_row_ptr[i]=0
    else:
        F_lower_row_ptr[i]=i-(n+1)
for i in range(size-(n+1)):
    if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
        F_lower_values[i]=-D_x/(dx**2.0)-Vx[i]/(2.0**dx)
 
F_lower=csr_matrix((F_lower_values, F_lower_col_indx, 
                     F_lower_row_ptr), shape=(size, size))

end1 = time.time()
print('matrix construction time: ',end1-start1,' s' )

matrix=(F_lower+F_upper+T_lower+T_medium+T_upper)
'''
dense_matrix = matrix.toarray()
output_file = 'matrix.txt'

# 打开文件以写入模式
with open(output_file, 'w') as file:
    # 获取矩阵的形状
    rows, cols = dense_matrix.shape
    
    # 遍历矩阵的每一行
    for i in range(rows):
        # 遍历矩阵的每一列
        for j in range(cols):
            # 将每个值格式化为双精度浮点数，保留小数点后三位
            value_string = '{:.15f}'.format(dense_matrix[i, j])
            
            # 写入值并添加间距
            file.write(value_string + ' ')
        
        # 写入换行符
        file.write('\n')

print("矩阵已成功写入文件。")
'''

RHS=np.zeros(size,dtype=Data_type)
for i in range(size):
    if (i+1)<=(n+1):
        RHS[i]=rhobulk
    else:
        if (i+1)%(n+1)!=0 and (i+1)%(n+1)!=1:
            RHS[i]=0.0
        elif (i+1)%(n+1)==0:
            RHS[i]=rhobulk  
        else: 
            RHS[i]=0.0

perfResult = benchmark(spilu,
                       (matrix,),
                       n_repeat=1)
#rho=spsolve(matrix,RHS)
lu=spilu(matrix,-10,(n+1))
rho=lu.solve(RHS)

filename2 = "ideal2Dxy.txt"
start2=time.time()
with open(filename2, 'w') as file:
    for i in range(size):
        if (i+1)%(n+1)!=0:
            
            value_string = '{:.15f}'.format(rho[i])
            file.write(str(value_string) + ' ')
        else:
            value_string = '{:.15f}'.format(rho[i])
            file.write(str(value_string) + ' ')
            file.write('\n')

end2=time.time()
print('file fill in time: ',end2-start2,' s' )
print(perfResult)
print('matrix size: ',size+1)
print('matrix solving time running on CPU: ',perfResult.cpu_times,' s')
print('matrix solving time running on CPU: ',perfResult.gpu_times,' s')
#print(rho[0])

