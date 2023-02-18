import argparse
from datetime import datetime
import yaml

def load_yaml(path):
    with open(path, 'r') as stream: 
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_config_and_time():
    print(f'Load input parameters...')
    # Input parameters.
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('--config_path', type=str, help='Input config path')
    args = parser.parse_args()

    # Load config.
    print(f'Load config and get datetime...')
    config = load_yaml(args.config_path)
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    return config, now