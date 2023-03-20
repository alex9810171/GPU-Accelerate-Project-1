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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device} to compute...')

    # Compute.
    A = torch.randn(2, 3, 4096, device=device)
    B = torch.randn(2, 3, 3, device=device)
    X = torch.linalg.solve(B,A)
    print(X)


if __name__ == '__main__':
    main()