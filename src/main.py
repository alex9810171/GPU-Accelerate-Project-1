import os, sys
import torch
from torch import nn
from torch.autograd.functional import jacobian
from src import utils

def main():
    # Load parameters & time.
    param, now = utils.get_config_and_time()

    # Create new result file directory.
    result_path = os.path.join(param['result_files']['path'], now)
    print(f'Create train file directory in {result_path}')
    os.makedirs(result_path, exist_ok=True)

    # Check gpu.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to compute...')

    # Compute.
    m = int(param['x_domain_size']/param['dx'])
    n = int(param['y_domain_size']/param['dy'])
    dradius = param['dy']


if __name__ == '__main__':
    main()