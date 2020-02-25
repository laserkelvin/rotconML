
import os

from src.pipeline import prepare_dataset, parse_calculations

"""
prepare_demo.py

This script prepares the datasets for a few hand picked molecules
that are supposed to be somewhat indicative of the performance
of the whole model.
"""


def main():
    #n_workers = os.getenv("NSLOTS", 4)
    n_workers = 4
    print(f"Using {n_workers} jobs.")
    print("Parsing the new calculation dataset.")
    parse_calculations.clean_data(
        data_dir="/data/sao/klee/projects/rotconml/data/demo-set/",
        interim_dir="/data/sao/klee/projects/rotconml/data/interim",
        maxatoms=30,
        nprocesses=n_workers,
        prefix="demo-"
    )
    print("Preprocessing data.")
    prepare_dataset.make_split_datasets(
        "/data/sao/klee/projects/rotconml/data/",
        prefix="demo-",
    )


if __name__ == "__main__":
    main()
