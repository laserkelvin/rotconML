"""
combine_dataset.py

This module contains the routines to put all of the data together into
a single output.

What's included here is a cookiecutter template for combining many
files into a single array with pseudocode.

Ideally, this will all turn it into HDF5 format which is much more
portable and managable than just NumPy memmap arrays for large datasets.

Requires Python >= 3.2
"""

from pathlib import Path
from typing import List, Type
from multiprocessing.pool import ThreadPool
import time

import h5py
import numpy as np
import pandas as pd
import periodictable as pt
import dask
from tqdm import tqdm
from dask import array as da
from dask_ml.decomposition import PCA as da_PCA
from dask_ml import preprocessing as da_pp
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from joblib import dump, load


def extract_rotcon_labels(mol_list: List):
    """
    Generate a NumPy array of labels from a list of Molecule objects.

    Parameters
    ----------
    mol_list : list
        List of Molecule objects generated from parse_g16
   
    Returns
    -------
    np.ndarray
        Array of labels for model interaction
    """
    labels = np.zeros((len(mol_list), 3), dtype=np.float)
    for index, molecule in enumerate(mol_list):
        try:
            label_set = [getattr(molecule, constant, 0) for constant in ["A", "B", "C"]]
        except AttributeError:
            print(f"{index} does not have rotational constants!")
        labels[index, :] = np.array(label_set, dtype=np.float)
    return labels


def store_datasets(h5_obj, storage_dict):
    """
    Function to store datasets into an h5py File object. This is primarily written
    to loopify the storing of training and validation datasets.
    
    The ordering of datasets should correspond with the `group_names` argument,
    for example:
        datasets=((X_train, Y_train), (X_test, Y_test))
    
    Parameters
    ----------
    h5_obj : h5py File object
        H5py object corresponding to an HDF5 file
    datasets : iterable
        Tuple of 2-tuple corresponding to pairs of split data
    group_names : list, optional
        List of the labels used to name the HDF5 groups, by default ["training", "validation"]
    """
    for group, pkg in storage_dict.items():
        group_h5 = h5_obj.create_group(group)
        labels = pkg["labels"]
        datasets = pkg["data"]
        for label, data in zip(labels, datasets):
            dataset_h5 = group_h5.require_dataset(
                label, shape=data.shape, dtype=data.dtype
            )
            da.store(data, dataset_h5)


def calculate_split_indexes(length: int, test_split=0.3, shuffle=True):
    """
    Helper function to shuffle and split an array via indexes. This
    function will generate a set of indexes with a specified length,
    optionally shuffling this array, and splitting the indexes into
    training and testing sets.
    
    Parameters
    ----------
    length : int
        Length of the index array
    test_split : float, optional
        Portion of the dataset to be used for testing, by default 0.3
    shuffle : bool, optional
        Whether or not to pre-scramble the indexes, by default True
    
    Returns
    -------
    train_indexes, test_indexes : np.ndarray
        The indexes corresponding to the training and testing sets
    """
    indexes = np.arange(length)
    # If we are shuffling the data set, scramble the indexes before
    # we divvy it up
    if shuffle is True:
        np.random.shuffle(indexes)
    # Work out the number of elements that will be in the test set
    test_length = int(test_split * indexes.size)
    test_indexes = indexes[:test_length]
    train_indexes = indexes[test_length:]
    return train_indexes, test_indexes


def custom_train_test_split(*arrays, test_split=0.3, shuffle=True):
    """
    Custom version of the dask/sklearn train_test_split function, specifically for
    splitting arrays up along the zeroth axis (i.e. the data points).
    
    Parameters
    ----------
    arrays : Iterable of np.ndarray
        [description]
    test_split : float, optional
        [description], by default 0.3
    shuffle : bool, optional
        [description], by default True
        
    Returns
    -------
    split_arrays : list of array-like
        List of arrays that have been divvied up into training and validation
        sets. Each array is returned in pairs of train/test; for example:
        
        X_train, X_test = custom_train_test_split(X)
    """
    train_indexes, test_indexes = calculate_split_indexes(
        arrays[0].shape[0], test_split, shuffle
    )
    split_arrays = list()
    # Loop over each array
    for array in arrays:
        array_pair = list()
        # Loop over training and testing indices
        for index in (train_indexes, test_indexes):
            array_pair.append(array[list(index)])
        split_arrays.append(array_pair)
    return split_arrays, train_indexes, test_indexes


def perform_PCA(
    X: np.ndarray, n_components=14, pca_obj=None, output_dir=None
) -> np.ndarray:
    """
    Function to perform the principal components reduction on a
    dataset. This is primarily for the eigenspectra, where we
    are trying to reduce the dimensionality of amount of information
    (which are mostly zeros anyway) needed to be encoded.
    
    This function will both determine the principal components, and
    perform the corresponding rotation of the input array into
    the principal components orientation.
    
    Parameters
    ----------
    X : np.ndarray
        A NumPy array of shape (n,m) where n is the number of datapoints,
        and m is the number of features
    n_components : int, optional
        Number of target components, by default 14

    Returns
    -------
    np.ndarray
        Dataset in principal components orientation, with shape (n,n_components)
    """
    if pca_obj is None:
        assert output_dir is not None
        pca_obj = PCA(n_components=n_components)
        _ = pca_obj.fit(X)
        dump(pca_obj, output_dir)
    else:
        pca_obj = load(pca_obj)
    return pca_obj.transform(X)


def standard_scaler(X: np.ndarray) -> np.ndarray:
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std


def matrix_minmax_scaler(X: np.ndarray) -> np.ndarray:
    X_scaled = X.copy()
    X_max = np.max(X, axis=1)
    X_scaled = X / X_max[:, None]
    return X_scaled


def one_scaler(X: np.ndarray) -> np.ndarray:
    """
    Divides each set of features by its corresponding maximum
    
    Parameters
    ----------
    X : np.ndarray
        [description]
    
    Returns
    -------
    np.ndarray
        [description]
    """
    maxes = np.max(X, axis=1)
    return X / maxes[:, None]


def encode_formula_vector(formula: str) -> np.ndarray:
    """
    Function for encoding a molecular formula into a vector of atomic
    composition.
    
    Parameters
    ----------
    formula : str
        Molecular formula as a string
    
    Returns
    -------
    np.ndarray
        A length-4 array with each position corresponding to the number
        of atoms of "H", "C", "O", and "N"
    """
    formula = pt.formula(formula)
    atoms = [getattr(pt.elements, atom) for atom in ["H", "C", "O", "N"]]
    encoding = np.zeros(len(atoms))
    for index, atom in enumerate(atoms):
        encoding[index] = formula.atoms.get(atom, 0)
    return encoding


def make_datasets(data_folder: str, prefix="", test_split=0.3, pca_obj=None):
    """
    Main driver function for getting data into a model-interaction ready
    level. This script follows the `clean_dataset.py` routines, which
    parses all the Gaussian output files and computes the Coulomb matrices.
    
    The labels are also created in this step, with the extract_rotcon_labels
    function.
    
    Parameters
    ----------
    data_folder : str
        String path to the data root folder
    """
    data_folder = Path(data_folder)
    interim_path = data_folder.joinpath("interim")
    out_path = data_folder.joinpath("processed")
    # This is the HDF5 file with the coulomb matrices
    coulomb_h5 = h5py.File(interim_path.joinpath(f"{prefix}rotconML-data.hd5"))
    # This is the HDF5 file for storing the model-ready data
    process_h5 = h5py.File(
        out_path.joinpath(f"{prefix}processed-rotconML-data.hd5"), mode="w"
    )
    print("Loading molecule table.")
    molecules = pd.read_pickle(interim_path.joinpath(f"{prefix}molecule-dataframe.pkl"))
    for column in ["u_A", "u_B", "u_C"]:
        molecules[column] = np.abs(molecules[column])
        # Make it so that the dipole moment is effectively boolean
        molecules.loc[molecules[column] > 0, column] = 1.0
    nmols = len(molecules)
    print(f"There are {nmols} entries in the dataset.")
    # This is a nested comprehension to get 3-tuple out of every
    # molecule object, corresponding to A, B, C rotational constants
    rotational_constants = molecules[["A", "B", "C"]].values
    rotational_constants = da.from_array(rotational_constants)
    three_pick = molecules[["A", "B", "C", "defect", "kappa"]].values
    three_pick = da.from_array(three_pick)
    six_pick = molecules[["A", "B", "C", "defect", "kappa", "u_A", "u_B", "u_C"]].values
    six_pick = da.from_array(six_pick)
    # Use Dask to store arrays into HDF5 format
    coulomb_array = da.from_array(coulomb_h5["coulomb_matrix"])
    scaled_coulomb_array = matrix_minmax_scaler(np.array(coulomb_h5["coulomb_matrix"]))
    scaled_coulomb_array = da.from_array(scaled_coulomb_array)
    # Take sort the eigenspectrum in descending magnitude
    eigen_array = np.sort(np.abs(coulomb_h5["eigenvalues"]))[:, ::-1]
    # Perform a PCA reduction on the eigenspectrum; from 30 to 12
    pca_eigen = perform_PCA(
        eigen_array, 12, pca_obj, out_path.joinpath(f"{prefix}pcamodel.pkl")
    )
    # Two methods of normalizing the PCA reduced eigenspectra
    norm_eigen = MaxAbsScaler().fit_transform(pca_eigen)
    minmax_eigen = MinMaxScaler().fit_transform(pca_eigen)
    # Four treatments of eigenvalues: full dimensional, PCA reduced, minmax scaled, and normalized to maximum
    eigen_array = da.from_array(eigen_array)
    pca_eigen = da.from_array(pca_eigen)
    minmax_eigen = da.from_array(minmax_eigen)
    norm_eigen = da.from_array(norm_eigen)

    encoding = np.zeros((nmols, 4))
    for index, row in molecules.iterrows():
        try:
            encoding[index, :] = encode_formula_vector(row["formula"])
        except ValueError:
            print(row["formula"])
    encoding = da.from_array(encoding)

    to_store = [
        coulomb_array,
        rotational_constants,
        eigen_array,
        three_pick,
        six_pick,
        pca_eigen,
        scaled_coulomb_array,
        encoding,
        minmax_eigen,
        norm_eigen,
    ]
    ############## Dataset splitting ##############
    # Use a custom version of the train_test_split function to split the
    # matrices/labels into train and validation sets
    pkg = custom_train_test_split(*to_store, test_split=test_split, shuffle=True)
    arrays, train_indexes, test_indexes = pkg
    # CM, constants, eig, three = arrays
    molecules.iloc[train_indexes].to_csv(
        out_path.joinpath(f"{prefix}training-dataframe.csv")
    )
    molecules.iloc[test_indexes].to_csv(
        out_path.joinpath(f"{prefix}validation-dataframe.csv")
    )
    # Prepare to organize the data
    storage_dict = {
        "training": {"data": [], "labels": []},
        "validation": {"data": [], "labels": []},
    }
    group_labels = [
        "coulomb_matrix",
        "rotational_constants",
        "eigenvalues",
        "fratsor",
        "rotcon_kd_dipoles",
        "pca_eigen",
        "norm_coulomb_matrix",
        "formula_encoding",
        "minmax_pca",
        "norm_pca",
        "functional_encoding"
    ]
    for pkg, label in zip(arrays, group_labels):
        for set_type, data in zip(["training", "validation"], pkg):
            storage_dict[set_type]["data"].append(data)
            storage_dict[set_type]["labels"].append(label)
    print(f"Storing datasets into HDF5 file. {str(out_path)}")
    store_datasets(process_h5, storage_dict)


def gather_data(
    h5_obj: Type[h5py._hl.files.File], key: str, indices: List[str], n_workers=8,
) -> da.core.Array:
    """
    Function to iterate through an entire set HDF5 file and gather up
    all of the data pertaining to a specified key, returning it as a
    Dask stacked array, such that a new axis is defined corresponding
    to each row of data. The collection step is parallelized with
    a `ThreadPool`. There is some overhead, but the speed up is actually
    non-negligble.

    For example, a key `rotational_constants` for every group in the
    h5py file is combined to yield an (n x m) array, where n is the
    number of groups, and m is the length of the dataset. The same
    can be extended to 2D arrays.
    
    Parameters
    ----------
    h5_obj : Type[h5py._hl.files.File]
        h5py File object containing the data
    key : str
        Key reference to the data to be collected
    
    Returns
    -------
    da.core.Array
        A stacked Dask array corresponding to one additional
        dimension, combining the dataset into a single array.
    """
    with ThreadPool(n_workers) as pool:
        arrays = pool.map(np.array, [h5_obj[index][key] for index in tqdm(indices)])
    return np.stack(arrays)


def dask_maxabs_scaler(dask_array: da.core.Array) -> da.core.Array:
    """
    In the same way as done in the sklearn `MaxAbsScaler` preprocessing
    function, this divides through an array 
    
    Parameters
    ----------
    dask_array : da.core.Array
        [description]
    
    Returns
    -------
    da.core.Array
        [description]
    """
    max_values = da.max(dask_array, axis=0)
    return dask_array / max_values


def eigen_pipeline(coulomb_eig: da.core.Array, n_components=12):
    """
    Perform the sequence of transformations on the Coulomb
    matrix eigenvalues. 
    
    Parameters
    ----------
    coulomb_eig : da.core.Array
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    # Perform a PCA reduction on the eigenvalues
    pca_obj = PCA(n_components)
    pca_obj.fit(coulomb_eig)
    pca_eig = pca_obj.transform(coulomb_eig)
    minmax_obj = MinMaxScaler()
    minmax_obj.fit(pca_eig)
    minmax_pca = minmax_obj.transform(pca_eig)
    # Use a custom function to perform the scaling
    norm_pca = MaxAbsScaler().fit_transform(pca_eig)
    return pca_eig, minmax_pca, norm_pca


def organize_network_split(
    n_entries: int,
    n_networks: int,
    h5_obj: Type[h5py._hl.files.File],
    df: Type[pd.DataFrame],
    shuffle=True,
    categorize_split=True,
):
    """
    Create a helper dictionary that organizes all of the data splitting.
    The fulll dataset is split twice - once into individual networks,
    and again on the subset for cross-validation.
    
    The dictionary contains references to the h5py groups, as well
    as the array indexes for the splitting.
    
    Parameters
    ----------
    n_entries : int
        [description]
    n_networks : 5
        [description]
    h5_obj : Type[h5py._hl.files.File]
        [description]
    shuffle : bool, optional
        [description], by default True
    
    Returns
    -------
    [type]
        [description]
    """
    indexes = np.arange(n_entries)
    if categorize_split is False:
        if shuffle:
            np.random.shuffle(indexes)
        split_indexes = np.split(indexes, n_networks)
    else:
        df["formula"] = df["formula"].astype(str)
        # Split into alkanes, oxygen-bearing, nitrogen-bearing, and ON bearing
        filtered = [
            df.loc[
                (~df["formula"].str.contains("O")) & (~df["formula"].str.contains("N"))
            ],
            df.loc[
                (~df["formula"].str.contains("N")) & (df["formula"].str.contains("O"))
            ],
            df.loc[
                (df["formula"].str.contains("N")) & (~df["formula"].str.contains("O"))
            ],
            df.loc[
                (df["formula"].str.contains("N")) & (df["formula"].str.contains("O"))
            ],
        ]
        split_indexes = [sliced_df.index.to_numpy() for sliced_df in filtered]
    org_dict = dict()
    for i, index_chunk in enumerate(split_indexes):
        # Sort the indexes into ascending order
        # index_chunk.sort()
        # This integer divide determines where to draw
        # the line for the training/validation split
        split_index = int(index_chunk.size * 0.7)
        # This makes sure that the data is properly scrambled
        # otherwise the validation set will always be the last
        # calculations performed within the group
        np.random.shuffle(index_chunk)
        org_dict[f"group{i}"] = {
            "ref": h5_obj.create_group(f"group{i}"),
            "indexes": index_chunk,
            "name": f"group{i}",
            "training_idx": index_chunk[:split_index],
            "validation_idx": index_chunk[split_index:],
            "training_length": len(index_chunk[:split_index]),
            "validation_length": len(index_chunk[split_index:]),
        }
    return org_dict


def make_qm9_datasets(data_folder: str, n_networks=5, n_workers=8, shuffle=True):
    """
    Main driver function for getting data into a model-interaction ready
    level. This script follows the `clean_dataset.py` routines, which
    parses all the Gaussian output files and computes the Coulomb matrices.
    
    The labels are also created in this step, with the extract_rotcon_labels
    function.
    
    The organization of the HDF5 file goes like this; each neural network model
    can independently access the training/validation datasets as before.
    root
    |
    |---> full
    |       |
    |       |--> rotational_constants
    |       |--> coulomb_eig
    |
    |---> group1
    |       |--> training
    |       |       |---> rotational_constants
    |       |--> validation
    |               |---> rotational_constants
    |---> group2
    |       |--> training
    |       |       |---> rotational_constants
    |       |--> validation
    |               |---> rotational_constants
    |...
    |---> groupn
    |       |--> training
    |       |       |---> rotational_constants
    |       |--> validation
    |               |---> rotational_constants
    
    Parameters
    ----------
    data_folder : str
        String path to the data root folder
    """
    # use threading for Dask operations
    dask.config.set(
        scheduler="threads", pool=ThreadPool(n_workers), num_workers=n_workers
    )
    data_folder = Path(data_folder)
    interim_path = data_folder.joinpath("interim")
    # This is the HDF5 file with parsed data
    interim_h5 = h5py.File(
        interim_path.joinpath("QM9_data.h5"), mode="r", libver="latest"
    )
    qm9_df = pd.read_csv(interim_path.joinpath("QM9_dataframe.csv"))
    out_path = data_folder.joinpath("processed")
    # This is the HDF5 file for storing the model-ready data
    process_h5 = h5py.File(
        out_path.joinpath("processed-QM9ML-data.hd5"), mode="w", libver="latest"
    )
    qm_dict_path = out_path.joinpath("processed-QM9-dict.pkl")
    array_dict = dict()
    # This is a nested comprehension to get 3-tuple out of every
    # molecule object, corresponding to A, B, C rotational constants
    indices = list(interim_h5.keys())
    indices.sort(key=int)
    full_h5 = process_h5.create_group("full")
    for key in tqdm(["rotational_constants", "coulomb_matrix", "coulomb_eig"]):
        last_time = time.time()
        print(f"Gathering up data for {key}.")
        array = gather_data(interim_h5, key, indices, n_workers)
        array_dict[key] = array
        dataset = full_h5.create_dataset(key, data=array, chunks=None)
        if key == "coulomb_eig":
            # Perform some processing on the coulomb matrix eigenvalues
            array_pipe = eigen_pipeline(array)
            for pipe, name in zip(array_pipe, ["pca_eigen", "minmax_pca", "norm_pca"]):
                dataset = full_h5.create_dataset(name, data=pipe)
                array_dict[name] = pipe
        print(f"Time taken: {time.time() - last_time}")
    # Get the formulas
    print("Generating formula encoding")
    formulas = [interim_h5[index].attrs.get("formula") for index in indices]
    encoding = np.zeros((len(formulas), 4), dtype=int)
    for index, formula in enumerate(formulas):
        encoding[index] = encode_formula_vector(formula)
    array_dict["formula_encoding"] = encoding
    # Store the formula stuffs
    full_h5.create_dataset("formula_encoding", data=encoding)
    print(f"Finished arranging data.")
    # Create a dictionary that organizes the data splitting
    org_dict = organize_network_split(
        len(interim_h5), n_networks, process_h5, qm9_df, shuffle
    )
    for subdict in org_dict.values():
        name = subdict["name"]
        group = subdict["ref"]
        print(f"Working on {name}.")
        # Loop over all of the datasets
        for subset in ["training", "validation"]:
            subgroup = group.require_group(subset)
            # Split the datasets. This is somewhat slow because we're transferring
            # between HDF5 matrices, but it does produce cleaner looking code.
            # If we want to speed this bit up, we'll have to reference the in-memory
            # arrays
            for key, dataset in full_h5.items():
                # Get the indexes to slice the data
                indexes = subdict.get(f"{subset}_idx")
                subgroup.create_dataset(key, data=array_dict[key][indexes])
            subgroup.create_dataset("indexes", data=indexes)
        # Delete the reference from org_dict, otherwise it makes
        # the dictionary unpickleable
        del subdict["ref"]
    dump(org_dict, qm_dict_path)


def make_split_datasets(data_folder: str, n_workers=8, shuffle=True, prefix=""):
    """
    Main driver function for getting data into a model-interaction ready
    level. This script follows the `clean_dataset.py` routines, which
    parses all the Gaussian output files and computes the Coulomb matrices.
    
    This routine will first aggregate all of the data into a "full" group, and
    subsequently divide it up into the four main categories according to
    `organize_network_split`; for now this is based on pure hydrocarbons,
    only oxygen, only nitrogen, and both oxygen and nitrogen containing
    species.
    
    The organization of the HDF5 file goes like this; each neural network model
    can independently access the training/validation datasets as before.
    root
    |
    |---> full
    |       |
    |       |--> rotational_constants
    |       |--> coulomb_eig
    |
    |---> group1
    |       |--> training
    |       |       |---> rotational_constants
    |       |--> validation
    |               |---> rotational_constants
    |---> group2
    |       |--> training
    |       |       |---> rotational_constants
    |       |--> validation
    |               |---> rotational_constants
    |...
    |---> groupn
    |       |--> training
    |       |       |---> rotational_constants
    |       |--> validation
    |               |---> rotational_constants
    
    Parameters
    ----------
    data_folder : str
        String path to the data root folder
    """
    # use threading for Dask operations
    dask.config.set(
        scheduler="threads", pool=ThreadPool(n_workers), num_workers=n_workers
    )
    data_folder = Path(data_folder)
    interim_path = data_folder.joinpath("interim")
    # This is the HDF5 file with parsed data
    interim_h5 = h5py.File(
        interim_path.joinpath(f"{prefix}rotconML-data.hd5"), mode="r",
    )
    mol_df = pd.read_pickle(interim_path.joinpath(f"{prefix}molecule-dataframe.pkl"))
    mol_df.reset_index(inplace=True, drop=True)
    out_path = data_folder.joinpath("processed")
    # This is the HDF5 file for storing the model-ready data
    process_h5 = h5py.File(
        out_path.joinpath(f"{prefix}processed-split-data.hd5"),
        mode="a",
    )
    qm_dict_path = out_path.joinpath(f"{prefix}processed-split-dict.pkl")
    array_dict = dict()
    # So this step is necessary because otherwise the indexing in the HDF5 file is
    # still treated as strings; i.e. 0, 1, 10, 100, 1000 instead of actual numeric
    # comparison. By working it out ahead of time we make sure all of the indexing
    # is the same
    indices = list(interim_h5.keys())
    indices.sort(key=int)
    full_h5 = process_h5.create_group("full")
    for key in tqdm(
        [
            "rotational_constants",
            "rotcon_kd_dipoles",
            "coulomb_matrix",
            "eigenvalues",
            "smi_encoding",
            "timeshift_eigenvalues",
            "functional_encoding",
            "eigenconcat"
        ]
    ):
        last_time = time.time()
        print(f"Gathering up data for {key}.")
        # Concatenate the eigenvalues and the spec constants
        if key == "eigenconcat":
            eigenvalues = gather_data(interim_h5, "eigenvalues", indices, n_workers)
            rotcon = gather_data(interim_h5, "rotcon_kd_dipoles", indices, n_workers)
            concat_set = np.concatenate([eigenvalues, rotcon], axis=1)
            dataset = full_h5.create_dataset(
                key,
                data=concat_set
            )
            array_dict[key] = concat_set
        else:
            array = gather_data(interim_h5, key, indices, n_workers)
            # Copy to a dictionary and reuse for later
            array_dict[key] = array
            dataset = full_h5.create_dataset(key, data=array, chunks=None)
            if key == "eigenvalues":
                # Perform some processing on the coulomb matrix eigenvalues
                array_pipe = eigen_pipeline(array)
                for pipe, name in zip(array_pipe, ["pca_eigen", "minmax_pca", "norm_pca"]):
                    dataset = full_h5.create_dataset(name, data=pipe)
                    array_dict[name] = pipe
        print(f"Time taken: {time.time() - last_time}")
    # Get the formulas
    print("Generating formula encoding")
    formulas = [
        str(interim_h5[str(index)].attrs.get("formula"), encoding="utf-8")
        for index in indices
    ]
    formula_encoding = np.zeros((len(formulas), 4), dtype=int)
    for index, formula in enumerate(formulas):
        formula_encoding[index] = encode_formula_vector(formula)
    array_dict["formula_encoding"] = formula_encoding
    # Store the formula stuffs
    full_h5.create_dataset("formula_encoding", data=formula_encoding)
    print(f"Finished arranging data.")
    # Create a dictionary that organizes the data splitting
    org_dict = organize_network_split(len(mol_df), 4, process_h5, mol_df, shuffle)
    for subdict in org_dict.values():
        name = subdict["name"]
        group = subdict["ref"]
        print(f"Working on {name}.")
        # Loop over all of the datasets
        for subset in ["training", "validation"]:
            subgroup = group.require_group(subset)
            # Split the datasets. This is somewhat slow because we're transferring
            # between HDF5 matrices, but it does produce cleaner looking code.
            # If we want to speed this bit up, we'll have to reference the in-memory
            # arrays
            for key, dataset in full_h5.items():
                # Get the indexes to slice the data
                indexes = subdict.get(f"{subset}_idx")
                subgroup.create_dataset(key, data=array_dict[key][indexes])
            subgroup.create_dataset("indexes", data=indexes)
        # Delete the reference from org_dict, otherwise it makes
        # the dictionary unpickleable
        del subdict["ref"]
    dump(org_dict, qm_dict_path)


if __name__ == "__main__":
    make_qm9_datasets("../../data")
