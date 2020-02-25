
import os
from pathlib import Path
from itertools import chain

import pandas as pd

from src.pipeline import prepare_dataset, parse_calculations


def init_rotamer_search(n_stable=20):
    interim_df = pd.read_pickle("../data/interim/newset-molecule-dataframe.pkl")
    # split up into the 20 most stable isomers for every formula
    sub_df = interim_df.groupby("formula").apply(lambda x: x.nsmallest(n_stable, "Etot"))
    sub_df.to_pickle("../data/interim/newset-rotamerinit-dataframe.pkl")


def main():
    n_workers = int(os.getenv("NSLOTS", 16))
    print(f"Using {n_workers} jobs.")
    print("Parsing the new calculation dataset.")
    # Accummulate all of the file paths
    # calc_dirs = ["newset", "raw", "newset-rotamers", "hydrocarbons"]
    # calc_dirs = ["newset"]
    # paths = [Path("/data/sao/klee/projects/rotconml/data").joinpath(path) for path in calc_dirs]
    # globbers = [path.rglob("*.log") for path in paths]
    # data_files = chain(*globbers)
    # parse_calculations.clean_data(
    #     data_dir="/data/sao/klee/projects/rotconml/data/newset/calcs",
    #     interim_dir="/data/sao/klee/projects/rotconml/data/interim",
    #     maxatoms=30,
    #     nprocesses=n_workers,
    #     prefix="newset-",
    #     data_files=list(data_files)
    # )
    #print("Preprocessing data.")
    # prepare_dataset.make_datasets(
    #     "/data/sao/klee/projects/rotconml/data/",
    #     prefix="newset-"
    # )
    print("Making split datasets.")
    prepare_dataset.make_split_datasets(
        "/data/sao/klee/projects/rotconml/data/",
        prefix="newset-"
    )


if __name__ == "__main__":
    main()
