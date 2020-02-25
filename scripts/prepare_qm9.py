
import os

from src.pipeline import prepare_dataset, parse_gdb


def main():
    n_workers = os.getenv("NSLOTS", 8)
    print(f"Using {n_workers} jobs.")
    print("Parsing QM9 dataset.")
    parse_gdb.batch_run(
        "/data/sao/klee/projects/rotconml/data/QM9",
        "/data/sao/klee/projects/rotconml/data/interim/",
        n_workers=n_workers,
    )
    print("Preprocessing data.")
    prepare_dataset.make_qm9_datasets(
        "/data/sao/klee/projects/rotconml/data/",
        n_workers=n_workers
    )
    
if __name__ == "__main__":
    main()
