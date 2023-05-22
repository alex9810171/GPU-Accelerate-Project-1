import os
from src import utils, test

def main():
    # Load parameters & time.
    param, now = utils.get_config_and_time()

    # Create new result file directory.
    result_path = os.path.join(param['result_files']['path'], 'matrix_inv')
    print(f'Create result file directory in {result_path}')
    os.makedirs(result_path, exist_ok=True)

    # Test some methods.
    test.matrix_inv(result_path, now)

if __name__ == '__main__':
    main()