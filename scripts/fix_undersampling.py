from typing import List

import h5py
import pandas as pd
import yaml
import numpy as np
from rdkit import Chem
from joblib import dump


n_duplicates = 10

def detect_substructs(x: "Mol", encodings: List[str]) -> np.ndarray:
    embedding = np.zeros(len(encodings), dtype=np.uint8)
    for index, encoding in enumerate(encodings):
        if x.HasSubstructMatch(encoding):
            embedding[index] += 1
    return embedding


mol_df = pd.read_pickle("../data/interim/newset-molecule-dataframe.pkl")

original_length = len(mol_df)

with open("encodings.yml") as read_file:
    encodings = yaml.safe_load(read_file)

# Create lookups with SMARTS
substructs = [Chem.MolFromSmarts(encoding) for encoding in encodings]
embeddings = (
    mol_df["functional"].apply(detect_substructs, args=(substructs,)).to_numpy()
)
# Necessary to get a 2D array out
embeddings = np.stack(embeddings, axis=0)
old_dict = dict()
for functional_group, counts in zip(encodings, embeddings.sum(axis=0)):
    old_dict[functional_group] = counts

# See notebook on how these indices were determined
# Use a NumPy mask to single out molecules that are NOT overrepresented
indices = np.where(
    (embeddings[:, 2] == 0)
    # & (embeddings[:, 4] == 0)
    & (embeddings[:, 10] == 0)
    & (embeddings[:, 11] == 0)
    & (embeddings[:, 17] == 0)
)[0]

# Duplicate the pandas DataFrame entries to get the indices
undersampled_df = mol_df.iloc[indices]
new_df = pd.concat([mol_df] + [undersampled_df] * n_duplicates)
# reset the indices
new_df.reset_index(drop=True, inplace=True)

new_embeddings = (
    new_df["functional"].apply(detect_substructs, args=(substructs,)).to_numpy()
)
new_embeddings = np.stack(new_embeddings, axis=0)
new_dict = dict()
for functional_group, counts in zip(encodings, new_embeddings.sum(axis=0)):
    new_dict[functional_group] = counts
print("New dataset composition:")
print(new_dict)

dump(
    {"old-composition": old_dict, "new-composition": new_dict},
    "../data/interim/augment-composition.pkl",
)

# Save the dataset as a pickle file
new_df.to_pickle("../data/interim/newset-augmented-dataframe.pkl")

# Redo the splitting into categories
hc_indices = new_df.loc[new_df["category"] == 0].index
hco_indices = new_df.loc[new_df["category"] == 1].index
hcn_indices = new_df.loc[new_df["category"] == 2].index
hcon_indices = new_df.loc[new_df["category"] == 3].index

group_indices = [hc_indices, hco_indices, hcn_indices, hcon_indices]

print(f"There are {hc_indices.size} hydrocarbons.")
print(f"There are {hco_indices.size} oxygen-bearing.")
print(f"There are {hcn_indices.size} nitrogen-bearing.")
print(f"There are {hcon_indices.size} both-bearing.")

# Read in the original arrays
original_arrays = h5py.File("../data/processed/newset-processed-split-data.hd5", "r")[
    "full"
]

# HDF5 file for writing the new data out to
new_arrays = h5py.File("../data/processed/newset-augmented-data.h5", "w")
full_set = new_arrays.create_group("full")
compositions = [new_arrays.create_group(f"group{index}") for index in range(4)]

array_dict = dict()
for label in [
    "rotcon_kd_dipoles",
    "eigenconcat",
    "eigenvalues",
    "formula_encoding",
    "functional_encoding",
    "timeshift_eigenvalues",
    "smi_encoding"
]:
    old_array = np.array(original_arrays[label])
    # This step is somewhat convoluted; we create a new array by
    # stacking tiled arrays based on the undersampled ones, along with
    # the original data
    new_array = np.concatenate(
        [old_array, np.tile(old_array[indices], (n_duplicates,) + (1,) * (old_array.ndim - 1))], axis=0
    )
    added_length = new_array.shape[0] - original_length
    # augment noise depending on the label
    if label == "rotcon_kd_dipoles":
        # rotational constants need different magnitude noise for rotational
        # constants and dipole moments, etc.
        noise = np.concatenate(
            [
                np.random.normal(scale=5.0, size=(added_length, 3)),
                np.random.normal(scale=0.2, size=(added_length, 5)),
            ],
            axis=-1,
        )
        new_array[original_length:] += noise
    # Store the full array
    _ = full_set.create_dataset(label, data=new_array)
    # Split the data into specific compositions
    for group_index, index_chunk in enumerate(group_indices):
        compositions[group_index].create_dataset(label, data=new_array[index_chunk])
