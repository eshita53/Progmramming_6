
import yaml
import argparse
from memory_profiler import memory_usage
import os

def get_config(file_name):
    with open(file_name, 'r', encoding="UTF-8") as stream:
        config = yaml.safe_load(stream)
    return config


def args_parser():
    """Parses command-line arguments for the file path"""
    parser = argparse.ArgumentParser(
        description=" Getting file path and fingerprint type")
    parser.add_argument('--file', type=str, required=True,
                        help="File path of config file")

    args = parser.parse_args()

    return args


def profile_memory_usage(func, *args, **kwargs):
    # Using memory_profiler to measure memory usage during execution
    mem_usage = memory_usage((func, args),
                             interval=0.01, max_usage=True)
    dataset = func(*args)
    return mem_usage, dataset


def append_to_csv(data, filename):
    try:
        # Open the file in append mode and without writing header
        data.to_csv(filename, mode='a', header=False, index=False)
        
    except FileNotFoundError:
        # If file doesn't exist, write the header
        data.to_csv(filename, mode='w', header=True, index=False)

def sub_classification(histology):
    if "Carcinoma" in histology:
        return 'Carcinoma'
    elif "Adenocarcinoma" in histology:
        return 'Adenocarcinoma'
    else:
        return 'Others'
    
def create_file_with_directory(filepath):
    directory = os.path.dirname(filepath)  # Extract directory path
    if directory and not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist
    
    # Create file if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            pass  # Create an empty file