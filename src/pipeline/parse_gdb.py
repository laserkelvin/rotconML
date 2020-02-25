"""
parse_gdb.py
Parsing routines for the QM9 database
"""

import re
from typing import Type, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import repeat

from tqdm import tqdm
from scipy import constants
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import numpy as np
import periodictable as pt
import h5py
import pandas as pd

from src.pipeline import utils


@dataclass
class GDBCalculation:
    index: int = 0
    rotational_constants: np.ndarray = np.zeros(3, dtype=float)
    coords: np.ndarray = np.empty(1)
    smi: str = ""
    natoms: int = 0
    mass: float = 0.0
    mass_array: np.ndarray = np.empty(1)
    atoms: np.ndarray = np.empty(1)
    coulomb_matrix: np.ndarray = np.empty(1)
    coulomb_eig: np.ndarray = np.empty(1)
    formula: str = ""

    @classmethod
    def from_xyz(cls, filepath: str):
        data_dict = parse_file(filepath)
        calc_obj = cls(**data_dict)
        return calc_obj

    def calculate_coulomb_matrix(self, max_atoms=30):
        """
        Calculate the Coulomb matrix associated with this molecule.
        
        Parameters
        ----------
        max_atoms : int, optional
            [description], by default 30
        """
        self.coulomb_matrix = utils.calc_coulomb(self.atoms, self.coords)
        self.coulomb_eig = np.sort(np.linalg.eig(self.coulomb_matrix)[0])[::-1]

    def calculate_rotational_constants(self):
        """
        Calculate the rotational constants of this molecule in units of MHz.
        This method will first work out the center of mass coordinates,
        and convert the masses and coordinates into SI units for the inertia
        tensor calculation. After diagonalizing, the principal moments of
        inertia are used to calculate the rotational constants in units of MHz.
        
        Returns
        -------
        abc, pmi, pm_axis
            1D NumPy arrays corresponding to the rotational constants in MHz,
            the principal moments of inertia (in kg m^2), and the principal
            moments vector
        """
        # Do the conversions into SI units; metres
        coords = self.coords * 1e-10
        masses = self.mass_array / constants.N_A
        # Calculate center of mass, and shift coordinates to COM
        com = utils.calculate_com(masses, coords)
        com_coords = coords - com
        # Calculate the principal moments and vector
        pmi, pm_axis = utils.calculate_inertia_tensor(masses, com_coords)
        # Calculate the rotational constants in units of MHz
        # Fudge factor is from Bernath, and for some reason there's a
        # factor 2 missing...
        abc = constants.h / (8.0 * np.pi ** 2 * constants.c * pmi)
        # Convert to MHz
        abc *= constants.c / 1e3
        abc.sort()
        self.rotational_constants = abc
        return abc, pmi, pm_axis

    def to_hdf5_group(self, hdf5_obj: Type[h5py.File]):
        """
        Function to export the current GDBCalculation class into
        an HDF5 format. The function requires reference to an `h5py`
        `File` object, and this class method will generate the
        corresponding hierarchy.

        Parameters
        ----------
        hdf5_obj : Type[h5py.File]
            Reference to an `h5py File`
        """
        group = hdf5_obj.require_group(str(self.index))
        # Get metadata
        for attribute in ["index", "smi", "natoms", "mass", "formula"]:
            value = getattr(self, attribute)
            if attribute in ["formula", "smi"]:
                value = value.encode("utf-8")
            group.attrs[attribute] = getattr(self, attribute)
        for item in [
            "coords",
            "mass_array",
            "atoms",
            "rotational_constants",
            "coulomb_matrix",
            "coulomb_eig",
        ]:
            group.create_dataset(item, data=getattr(self, item))

    def __repr__(self):
        return f"Calculation: {self.index}"

    def rdkit_pipeline(self):
        """
        Use RDKit to run through some chemical diagnositics.
        The function uses the parsed SMILES string to create a molecule
        object in RDKit, and then runs through a few functions
        to determine number of bonds, aromaticity, rings, and
        functional groups.        
        """
        chem_dict = dict()
        # get ring information
        rd_mol = Chem.MolFromSmiles(self.smi)
        if rd_mol is not None:
            chem_dict["formula"] = CalcMolFormula(rd_mol)
            rings = rd_mol.GetRingInfo()
            chem_dict["n_rings"] = rings.NumRings()
            ring_atoms = 0
            # Atomic information
            for atom in rd_mol.GetAtoms():
                if atom.GetIsAromatic() is True:
                    chem_dict["aromatic"] = True
                if atom.IsInRing() is True:
                    ring_atoms += 1
            chem_dict["n_ring_atoms"] = ring_atoms
            # Work out statistics of bond orders
            mapping = dict()
            # index runs over number of bonds, bond_type is the name
            for index, bond_type in zip(
                range(1, 4), ["n_single_bond", "n_double_bond", "n_triple_bond"]
            ):
                chem_dict[bond_type] = 0
                mapping[index] = bond_type
            for bond in rd_mol.GetBonds():
                bond_order = int(bond.GetBondTypeAsDouble())
                chem_dict[mapping[bond_order]] += 1
            # Compute the average bond order
            orders = [
                chem_dict[bond_type]
                for bond_type in ["n_single_bond", "n_double_bond", "n_triple_bond"]
            ]
            weights = [1, 2, 3]
            chem_dict["avg_bo"] = np.average(orders, weights=weights)
            # Matching functional groups
            smrts = {
                "carbonyl": "[CX3]=[OX1]",
                "amine": "[NX3;H2,H1;!$(NC=O)]",
                "alcohol": "[OX2H]",
                "enol": "[OX2H][#6X3]=[#6]",
                "peroxide": "[OX2,OX1-][OX2,OX1-]",
                "nitrile": "[NX1]#[CX2]",
                "vinyl": "[$([CX3]=[CX3])]",
                "allene": "[$([CX2](=C)=C)]",
                "aldehyde": "[CX3H1](=O)[#6]",
                "ketone": "[#6][CX3](=O)[#6]",
                "ether": "[OD2]([#6])[#6]",
            }
            for key, smrt in smrts.items():
                rd_smrt = Chem.MolFromSmarts(smrt)
                chem_dict[key] = len(rd_mol.GetSubstructMatch(rd_smrt))
            self.__dict__.update(**chem_dict)


def read_coords(content: str) -> [np.ndarray, np.ndarray]:
    """
    [summary]
    
    Parameters
    ----------
    content : str
        [description]
    
    Returns
    -------
    [np.ndarray, np.ndarray]
        [description]
    """
    coord_regex = re.compile(r"[CNOH]\s+-?\d.\d+\s+-?\d.\d+\s+-?\d.\d+\s+-?\d.\d+")
    match = coord_regex.findall(content)
    coords = np.zeros((len(match), 3), dtype=float)
    masses = np.zeros(len(match))
    atoms = np.zeros(len(match))
    for index, row in enumerate(match):
        split_row = row.split("\t")
        split_row = [value.lstrip() for value in split_row]
        atom = getattr(pt.elements, split_row[0])
        coords[index] = [float(value) for value in split_row[1:4]]
        masses[index] = atom.mass
        atoms[index] = atom.number
    return coords, masses, atoms


def parse_file(filepath: str) -> Dict[str, Any]:
    """
    [summary]
    
    Parameters
    ----------
    filepath : str
        [description]
    
    Returns
    -------
    Dict[str, Any]
        [description]
    """
    with open(filepath, "r") as read_file:
        contents = read_file.read()
    coords, masses, atoms = read_coords(contents)
    total_mass = masses.sum()
    contents = contents.split("\n")
    # Get the number of atoms
    natoms = int(contents[0])
    # Split up the second line which contains all the calculated values
    calc_line = contents[1].split("\t")
    index = int(contents[1].split()[1])
    # Convert to MHz for rotational constants
    rotational_constants = [float(value) * 1000.0 for value in calc_line[2:5]]
    smi = contents[-3].split("\t")[-2]
    data_dict = {
        "mass": total_mass,
        "mass_array": masses,
        "natoms": natoms,
        "smi": smi,
        "coords": coords,
        "index": index,
        "rotational_constants": rotational_constants,
        "atoms": atoms,
    }
    return data_dict


def process_data(filepath: str, natoms=50) -> Type[GDBCalculation]:
    """
    This is the general function used to perform parsing and calculations
    of the QM9 dataset. The XYZ file is read in, the rotational constants
    are calculated to higher precision, the Coulomb matrix and its eigen
    values are calculated, and finally some statistics on 
    
    Parameters
    ----------
    filepath : str
        [description]
    natoms : int, optional
        [description], by default 50
    
    Returns
    -------
    [type]
        [description]
    """
    molecule = GDBCalculation.from_xyz(filepath)
    # _ = molecule.calculate_rotational_constants()
    molecule.calculate_coulomb_matrix(natoms)
    molecule.rdkit_pipeline()
    return molecule


def batch_run(xyz_path: str, interim_path: str, n_workers=8, natoms=50):
    """
    This function automates the whole processing of the QM9 database
    into two sets of data; a Pandas dataframe for interactive analysis,
    and a HDF5 file for subsequent machine learning.
    
    Parameters
    ----------
    xyz_path : str
        [description]
    interim_path : str
        [description]
    n_workers : int, optional
        [description], by default 8
    natoms : int, optional
        [description], by default 50
    """
    interim_path = Path(interim_path)
    h5_path = interim_path.joinpath("QM9_data.h5")
    df_path = interim_path.joinpath("QM9_dataframe.csv")
    h5_file = h5py.File(h5_path, mode="w")
    xyz_path = Path(xyz_path)
    # Format for dataframe dumping
    molecule_data = list()
    to_drop = [
        "rotational_constants",
        "coords",
        "coulomb_matrix",
        "coulomb_eig",
        "mass_array",
        "atoms",
    ]
    # Parallelize the parsing
    with ProcessPoolExecutor(n_workers) as executor:
        for molecule in executor.map(
            process_data, tqdm(xyz_path.rglob("*.xyz")), repeat(natoms)
        ):
            molecule.to_hdf5_group(h5_file)
            A, B, C = molecule.rotational_constants
            data_dict = molecule.__dict__
            # Drop some of the columns that don't fit into a dataframe
            for key in to_drop:
                del data_dict[key]
            for key, value in zip(["A", "B", "C"], [A, B, C]):
                data_dict[key] = value
            molecule_data.append(data_dict)
    df = pd.DataFrame(molecule_data)
    # Just formatting
    df.loc[df["aromatic"] != True, "aromatic"] = False
    df.to_csv(df_path, index=False)


if __name__ == "__main__":
    batch_run(
        "/data/sao/klee/projects/rotconml/data/QM9",
        "/data/sao/klee/projects/rotconml/data/interim/",
        n_workers=8,
    )
