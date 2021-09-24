import argparse
from pyaudioann.data import generate_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "data_manager.py",
        description="Script to handle data management",
    )
    parser.add_argument(
        "-g",
        "--generate",
        action="store_true",
        dest="generate",
        help="""Generates output samples""",
    )
    results = parser.parse_args()
    
    if results.generate:
        generate_datasets()