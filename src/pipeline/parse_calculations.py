"""
clean_dataset.py

This module will contain routines for cleaning and formatting
the data ready for training, testing, and analysis.

Ideally, this will all turn it into HDF5 format which is much more
portable and managable than just NumPy memmap arrays for large datasets.
"""

from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor

from src.pipeline import utils

from joblib import dump
from tqdm import tqdm
from rdkit import Chem
import yaml
import h5py
import pandas as pd
import numpy as np
import numba
import periodictable as pt


def log2data(filepath: Path, maxatoms: int):
    """
    Function to extract the juicy data from a Gaussian log
    file. Creates a Molecule object from the logfile,
    and will also compute the associate Coulomb matrix
    """
    molecule = utils.parse_g16(filepath)
    return molecule


def encode_functional_groups(smi, search_objs):
    if type(smi) == str:
        smi = Chem.MolFromSmiles(smi)
    encoded = list()
    for search_obj in search_objs:
        if smi.HasSubstructMatch(search_obj):
            encoded.append(1)
        else:
            encoded.append(0)
    return np.array(encoded)


@numba.njit
def calc_coulomb(charges: np.ndarray, coords: np.ndarray, coulomb_mat: np.ndarray):
    """
    Function to compute to Coulomb matrix based on a set
    of charges and XYZ coordinates.

    This calculation is performed with LLVM code because
    it can be - it doesn't seem to appreciably speed things
    up more so than the outer loop with the multiprocessing,
    but it's good to future proof for larger matrices.

    Parameters
    ----------
    charges : np.ndarray
        Array of atomic charges/numbers; length maxatoms
    coords : np.ndarray
        Array of XYZ coordinates; shape (maxatoms, 3)
    coulomb_mat : np.ndarray
        Reference to the Coulomb matrix array; this should have
        dimensions maxatoms x maxatoms
    """
    for i in range(coords.shape[0]):
        # Loop over second atom
        for j in range(coords.shape[0]):
            if i == j:
                value = 0.5 * charges[i] ** 2.4
            else:
                # Calculate Euclidean distance between points
                dist = np.linalg.norm(coords[i, :] - coords[j, :])
                value = (charges[i] * charges[j]) / dist
            coulomb_mat[i, j] = value


def xyz2coulombmat(coord_mat: np.ndarray, maxatoms: int) -> np.ndarray:
    """
    Routine to calculate the Coulomb matrix for a given set of
    Cartesian coordinates. The formatting of the coordinates
    must include the atomic charge as the first column, followed
    by x,y,z.
    
    Initialization is done outside of the calculation so that
    the actual loop is done with Numba njit'd code.
    
    Parameters
    ----------
    coord_mat : np.ndarray
        NumPy 2D array of shape (n,4), where each row corresponds to an atom.
        First column corresponds to molecular charge, and the
        remaining three are the x,y,z coordinates.
    maxatoms : int
        Maximum number of atoms in the Coulomb matrix; used
        to create a nxn zero matrix
    
    Returns
    -------
    coulomb_mat : np.ndarray
        NumPy array for the Coulomb matrix
    """
    # First column is the atomic charge
    charges = coord_mat[:, 0]
    coords = coord_mat[:, 1:]
    # initialize the Coulomb matrixx
    coulomb_mat = np.zeros((maxatoms, maxatoms), dtype=np.float)
    # Loop over first atom
    calc_coulomb(charges, coords, coulomb_mat)
    return coulomb_mat


@np.vectorize
def formula2mass(formula: str) -> float:
    """
    Vectorized function that will calculate the mass of a given
    molecular formula. The function converts a formula string
    into a `periodictable.Formula` object, and returns the
    mass attribute.
    
    Parameters
    ----------
    formula : str
        Molecular formula
    
    Returns
    -------
    float
        Mass of the molecule in amu
    """
    formula_obj = pt.formula(formula)
    return formula_obj.mass


def atom_featurization(atom: object) -> List[int]:
    """
    Function to compute the feature vector for a given atom.
    This method works specifically for atoms, and not the other
    symbols that are part of the SMILES encoding.
    This is based on a re-write of the code obtained from http://www.dna.bio.keio.ac.jp/smiles/,
    which is from: 

    Hirohara et al., Convolutional Neural Network Based on SMILES 
    Representation of Compounds for Detecting Chemical Motif. 
    BMC Bioinformatics 2018, 19 (19), 526. 
    https://doi.org/10.1186/s12859-018-2523-5.

    Parameters
    ----------
    atom : rdkit `Atom` object
        rdkit `Atom` object, part of a `Molecule` object.

    Returns
    -------
    List[str]
        Encoded feature vector
    """
    # Dictionary map for each of the chriality/hybridization types
    chiral_dict = {
        "CHI_UNSPECIFIED": 0,
        "CHI_TETRAHEDRAL_CW": 1,
        "CHI_TETRAHEDRAL_CCW": 2,
        "CHI_OTHER": 3,
    }
    hybrid_dict = {
        "UNSPECIFIED": 0,
        "S": 1,
        "SP": 2,
        "SP2": 3,
        "SP3": 4,
        "SP3D": 5,
        "SP3D2": 6,
        "OTHER": 7,
    }
    if atom.GetSymbol() == "H":
        feature = [1, 0, 0, 0, 0]
    elif atom.GetSymbol() == "C":
        feature = [0, 1, 0, 0, 0]
    elif atom.GetSymbol() == "O":
        feature = [0, 0, 1, 0, 0]
    elif atom.GetSymbol() == "N":
        feature = [0, 0, 0, 1, 0]
    else:
        feature = [0, 0, 0, 0, 1]

    # Encode several parameters about the SMILES string
    feature.append(atom.GetTotalNumHs() / 8)
    feature.append(atom.GetTotalDegree() / 4)
    feature.append(atom.GetFormalCharge() / 8)
    feature.append(atom.GetTotalValence() / 8)
    feature.append(atom.IsInRing() * 1)
    feature.append(atom.GetIsAromatic() * 1)

    # Encode chirality
    f = [0] * (len(chiral_dict) - 1)
    if chiral_dict.get(str(atom.GetChiralTag()), 0) != 0:
        f[chiral_dict.get(str(atom.GetChiralTag()), 0)] = 1
    feature.extend(f)

    f = [0] * (len(hybrid_dict) - 1)
    if hybrid_dict.get(str(atom.GetHybridization()), 0) != 0:
        f[hybrid_dict.get(str(atom.GetHybridization()), 0)] = 1
    feature.extend(f)

    return feature


def neu_atom_featurization(symbol: str) -> np.ndarray:
    """
    Stripped down version of the SMILES atom encoding; this
    only encodes the atom, and not the actual chemical
    representation....
    
    Parameters
    ----------
    symbol : str
        [description]
    
    Returns
    -------
    np.ndarray
        [description]
    """
    encoding = np.zeros(7, dtype=int)
    if symbol.isupper():
        offset = 3
    else:
        offset = 0
    if symbol == "H":
        encoding[0] += 1
    else:
        mapping = ["C", "O", "N"]
        index = mapping.index(symbol.upper())
        encoding[index + offset + 1] += 1
    return encoding


def character_featurization(
    value: str, charge_flag: int, labels: List[str]
) -> List[int]:
    """
    Function to convert a non-atomic symbol into a vector representation.
    
    Parameters
    ----------
    value : str
        [description]
    
    Returns
    -------
    List[int]
        [description]
    """
    encoding = np.zeros(shape=21, dtype=int)
    if value.isdigit():
        # This section encode ring structures; index 19 and 20
        # mark the beginning and end of rings
        if charge_flag == 0:
            if value in labels:
                encoding[20] += 1
            else:
                labels.append(value)
                encoding[19] += 1
        # This part encodes ionic charge; flag is set
        # to 1 when a positive or negative sign is encountered
        else:
            encoding[int(value) - 1 + 12] += 1
            charge_flag = 0
    else:
        map_list = ["(", ")", "[", "]", ".", ":", "=", "#", "\\", "/", "@", "+", "-"]
        encoding[map_list.index(value)] += 1
        # If charged, flag it so the next step is to read the charge
        if value in ["+", "-"]:
            charge_flag += 1
    return encoding, charge_flag, labels


def calculate_kappa(A: float, B: float, C: float) -> float:
    """
    Calculate Ray's asymmetry parameter.
    
    Parameters
    ----------
    A : float
        [description]
    B : float
        [description]
    C : float
        [description]
    
    Returns
    -------
    float
        [description]
    """
    return (2 * B - A - C) / (A - C)


def featurize_smiles(smi: str, maxchar=100) -> np.ndarray:
    """
    Convert a SMILES code string into a vector representation.
    
    Parameters
    ----------
    smi : str
        [description]
    maxchar : int, optional
        Maximum length of the SMILES coding, by default 100
        This correspond to the maximum length of SMILES strings that
        will be considered in the study, which is used to initialize
        the encoding array
    
    Returns
    -------
    np.ndarray
        NumPy 2D array; rows correspond to the SMILES character,
        with each column encoding the corresponding character
    """
    # Rows correspond to a SMILES string character, and columns
    # represent the encoding
    encoding = np.zeros(shape=(maxchar, 28), dtype=int)
    # molecule = Chem.MolFromSmiles(smi)
    charge_flag = 0
    # Atom indices must be kept track of separately since the rdkit Molecule
    # does not line up with the SMILES enumeration
    atom_index = 0
    labels = list()
    specials = ["(", ")", "[", "]", ".", ":", "=", "#", "\\", "/", "@", "+", "-"]
    for index, value in enumerate(smi):
        if value.isalpha():
            # atom = molecule.GetAtomWithIdx(atom_index)
            encoding[index, :7] = neu_atom_featurization(value)
            atom_index += 1
        elif value in specials:
            char_encode, charge_flag, labels = character_featurization(
                value, charge_flag, labels
            )
            encoding[index, 7:] = char_encode
    return encoding


@numba.jit
def onehot_smiles(smi: str, maxchar=100) -> np.ndarray:
    """
    Simpler one hot encoding for the SMILES strings. Iterates over a string
    of SMILES, and uses a list to lookup what the index of the encoding
    should be.
    
    Parameters
    ----------
    smi : str
        A string of SMILES
    maxchar : int, optional
        Maximum number of characters to zero pad the array by; by default 100
    
    Returns
    -------
    np.ndarray
        A NumPy 1D array, with each element corresponding to the index used
        to encode the characters
    """
    smi_list = [
        "H",
        "C",
        "O",
        "N",
        "c",
        "o",
        "n",
        "(",
        ")",
        "[",
        "]",
        ".",
        ":",
        "=",
        "#",
        "\\",
        "/",
        "@",
        "+",
        "-",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    smi_array = np.zeros((maxchar, len(smi_list) + 1), dtype=np.int32)
    for row_index, char in enumerate(smi):
        # There is a shift of 1 because I'm reserving zero for
        # garbage; i.e. the absence of a character
        encoding_index = smi_list.index(char) + 1
        smi_array[row_index, encoding_index] += 1
    smi_array[len(smi_list) :, 0] += 1
    return smi_array


# @numba.njit
def timeshift_array(array: np.ndarray, window_length=4, n_timeshifts=100) -> np.ndarray:
    """
    Uses a neat NumPy trick to vectorize a sliding operation on a 1D array.
    Basically uses a 2D indexer to generate n_shifts number of windows of
    n_elements length, such that the resulting array is a 2D array where
    each successive row is shifted over by one.
    
    The default values are optimized for a maximum of 30 atoms.
    
    This is based off this SO answer:
    https://stackoverflow.com/a/42258242
    
    Parameters
    ----------
    array : np.ndarray
        [description]
    n_elements : int
        Length of each window, by default 5
    n_timeshifts : int, optional
        Number of timeslices, by default 100
    
    Returns
    -------
    np.ndarray
        NumPy 2D array with rows corresponding to chunks of a sliding
        window through the input array
    """
    shifted = np.zeros((n_timeshifts, window_length), dtype=np.float32)
    n_actual = array.size - window_length + 1
    indexer = np.arange(window_length).reshape(1, -1) + np.arange(n_actual).reshape(
        -1, 1
    )
    shifted[:n_actual, :] = array[indexer]
    return shifted


def clean_data(
    data_dir: str,
    interim_dir: str,
    maxatoms: int,
    nprocesses=8,
    prefix="",
    data_files=None,
):
    """
    Function to process all of the logfiles in parallel, and extract
    the Coulomb matrix and Molecule objects.

    The matrices are saved in the data/interim/rotconML-data.hd5 file,
    as a 3D matrix; the first index corresponds to the molecule.
    
    Parameters
    ----------
    data_dir : str
        Root directory containing all of the calculations. The globbing occurs
        one level down from this, i.e. */*.log and so ensure the path pointed
        to includes this.
    interim_dir : str
        Filepath to save the preprocessed data to
    maxatoms : int
        Number of atoms to zero the Coulomb matrices with
    nprocesses : int, optional
        Number of processes to parallelize with, by default 8
    prefix : str, optional
        Prefix used to prepend serialized datasets for identification, by default ""
    """
    print(f"Aggregating data from {data_dir}.")
    data_dir = Path(data_dir)
    interim_dir = Path(interim_dir)
    # Get all of the logfiles and sort them; this makes the globbing
    # system independent
    if data_files is None:
        data_files = list(data_dir.rglob("*/*.log"))
        data_files = sorted(data_files)
    print(f"Initializing HDF5 file at {interim_dir}/{prefix}rotconML-data.hd5")
    # Prepare for data dumping
    h5_obj = h5py.File(interim_dir.joinpath(f"{prefix}rotconML-data.hd5"), mode="w")
    print(f"Working on parsing molecules.")
    molecules = list()
    n_items = len(data_files)
    print(f"There are {n_items} calculations to process.")

    def atom_repeater(maxatoms):
        while True:
            yield maxatoms

    # Work in parallel, retrieving all the successful calculations
    with ProcessPoolExecutor(nprocesses) as executor:
        for index, molecule in tqdm(
            enumerate(executor.map(log2data, data_files, atom_repeater(maxatoms))),
            total=n_items,
        ):
            # Check that the calculation is successful
            if molecule.success is True:
                molecules.append(molecule)
            else:
                # Make an exception for when it's basically converged
                if abs(molecule.opt_delta) <= 1e-12:
                    molecules.append(molecule)
    # The remaining operations sanitize the dataset
    df = pd.DataFrame([molecule.__dict__ for molecule in molecules])
    # Take the absolute value of dipole moment
    for col in ["u_A", "u_B", "u_C"]:
        df[col] = np.abs(df[col])
    # use the rotational constants to determine if the molecule is unique
    # rounded to the nearest decimal, it's extremely unlikely two molecules
    # have the same A,B,C and dipole moments
    n_original = len(df)
    df = df[~df[["A", "B", "C", "u_A", "u_B", "u_C"]].round(0).duplicated()]
    print(f"Removed {n_original - len(df)} duplicates.")
    # Drop missing A,B,C
    df = df.loc[df[["A", "B", "C"]].sum(axis=1) != 0.0]
    # Take only singlets
    df = df.loc[df["multi"] == 1]
    # Drop transition states
    df["harm_freq"] = df["harm_freq"].apply(lambda x: np.array(x))
    mask = df["harm_freq"].apply(lambda x: np.any(x <= 0.0))
    df = df.loc[~mask]
    # drop the complexes, or dissociated molecules
    # the old method was based on crazy inertial defects
    df = df.loc[df["fragments"] == False]
    print("Calculating masses")
    df["mass"] = formula2mass(df["formula"])
    # Recalculate the inertial defect, since Gaussian is wrong?
    df.loc[:, "defect"] = df.apply(utils.calc_inertial_defect, axis="columns")
    # Drop nans
    df.dropna(
        subset=["A", "B", "C", "kappa", "defect", "u_A", "u_B", "u_C"], inplace=True
    )
    df["kappa"] = calculate_kappa(df["A"], df["B"], df["C"])
    df.reset_index(inplace=True)
    # categorize the molecules according to composition
    masks = [
        (~df["formula"].str.contains("O")) & (~df["formula"].str.contains("N")),
        (df["formula"].str.contains("O")) & (~df["formula"].str.contains("N")),
        (~df["formula"].str.contains("O")) & (df["formula"].str.contains("N")),
        (df["formula"].str.contains("O")) & (df["formula"].str.contains("N")),
    ]
    for index, mask in enumerate(masks):
        df.loc[mask, "category"] = index
    print(f"Saving molecules as a dataframe.")
    # prepare to work on the functional group encodings
    with open("encodings.yml") as read_file:
        encodings = yaml.safe_load(read_file)
    # Generate substructure search objects
    group_searches = [Chem.MolFromSmarts(encoding) for encoding in encodings]
    df["functional"] = df["smi"].apply(Chem.MolFromSmiles)
    # Convert the SMILES into `Chem` objects, and wherever that fails drop it
    df.dropna(subset=["functional"], inplace=True)
    df.to_pickle(interim_dir.joinpath(f"{prefix}molecule-dataframe.pkl"))
    # Initialize a dataset for holding the Coulomb matrix
    # Organize array data types like the Coulomb matrix and Cartesian
    # coordinates into HDF5
    print("Arranging matrices.")
    # Load the SMARTS encodings
    for index, row in tqdm(df.iterrows(), total=len(df)):
        sub_group = h5_obj.create_group(f"{index}")
        # Store information about each molecule as attributes
        for key in ["smi", "formula", "mass"]:
            value = getattr(row, key, "empty")
            if key != "mass":
                value = row[key].encode("utf-8")
            sub_group.attrs[key] = value
        # Make sure the coordinates are all floats, redundant
        coords = row["coords"].astype(float)
        coulomb = xyz2coulombmat(coords, maxatoms)
        eigenvalues = np.abs(np.linalg.eigvals(coulomb))
        eigenvalues.sort()
        eigenvalues = eigenvalues[::-1]
        sub_group.create_dataset("coulomb_matrix", data=coulomb)
        sub_group.create_dataset("eigenvalues", data=eigenvalues)
        # Store rotational constants and whathaveyou
        sub_group.create_dataset(
            "rotational_constants", data=row[["A", "B", "C"]].to_numpy().astype(float),
        )
        sub_group.create_dataset(
            "rotcon_kd_dipoles",
            data=row[["A", "B", "C", "kappa", "defect", "u_A", "u_B", "u_C"]]
            .to_numpy()
            .astype(float),
        )
        sub_group.create_dataset("cartesian_coordinates", data=coords)
        # Calculate the timeshifted eigenvalues for LSTM
        timeshift_eigen = timeshift_array(eigenvalues)
        sub_group.create_dataset("timeshift_eigenvalues", data=timeshift_eigen)
        # Encode SMILES
        smi_encoding = onehot_smiles(row["smi"], maxchar=100)
        sub_group.create_dataset("smi_encoding", data=smi_encoding)
        # Encode functional groups
        functional_encoding = encode_functional_groups(row["smi"], group_searches)
        sub_group.create_dataset("functional_encoding", data=functional_encoding)


if __name__ == "__main__":
    clean_data(
        data_dir="/data/sao/klee/projects/rotconml/data/newset/calcs",
        interim_dir="/data/sao/klee/projects/rotconml/data/interim",
        maxatoms=30,
        nprocesses=8,
    )
