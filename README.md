# rotconML

## Identifying molecules with probabilistic deep learning

---

# Aim

This project builds a series of deep learning models to help identify molecules
based on their rotational spectroscopic parameters.

# Context

- Spectral features from rotational spectra fit to rotational Hamiltonians
- Spectroscopic parameters alone do not uniquely identify a molecule
- Molecular identification usually done by matching parameters to molecular structure
- Parameters are highly encoded: chemical and structural information is deeply embedded and not easily retrieved

# Solution

This project implements a series of deep learning models that map spectroscopic
parameters to identifying information about a molecule. In order to ensure that
the full breadth of possible structures are explored, the models are constructed
in a probabilistic context using dropout layers as an approximation to Bayesian
sampling.

---

# File structure details

## `src`

This folder contains all of the backend Python code. In the latest iteration, I
used PyTorch more or less exclusively, and you can find that under
`src.models.torch_models`.

The `pipeline` module contains all of the routines I wrote to perform data
cleaning and formatting for analysis. The code basically parses Gaussian 16
output files, and extracts all of the relevant data and puts them into HDF5
files for analysis.

The `visualzation` module contains a few quick routines for plotting results,
which I used early on for analyzing performance. For figure making I did not
rely on these routines as much.

## `models`

This folder, specifically `models/torch/`, is where all the model training is
performed. These scripts utilize GPUs to train the models, and the `wandb`
Python package to track experiments. The two subfolders, `tensorflow` and
`torch`, are implementations with those respective libraries.

## `production`

This is where  the demonstrations were done after the models are trained. There
is a script, `unified_model_test.py`, which shows how the models can be used.

## `scripts`

This folder is where the preprocessing is done. The scripts will do all of the
relevant data parsing, and put data into the right locations for subsequent
model training and analysis. The two main scripts are `prepare_newset.py` and
`prepare_demo.py`; the former generates the datasets for the main bulk of the
work and the latter is for demonstration purposes.

A more recent and important script is `fix_undersampling.py`, which does what
the number suggests. This script will check all of the `newset` dataset entries
and perform SMARTS substructure searches to determine which functional groups
are undersampled, and uses this information to augment the final dataset by
boosting creating noise-perturbed copies of existing examples.

## License

rotconML - a project on probabilistic molecule identification with PyTorch 

Copyright (C) 2019-2020 Kin Long Kelvin Lee

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
