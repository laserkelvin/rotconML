from src.models.torch_models import ChainModel
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
from joblib import dump
from tqdm import tqdm


data = h5py.File("../data/processed/demo-processed-split-data.hd5", mode="r")["full"]
constants = np.array(data["rotcon_kd_dipoles"])
target_spectra = data["eigenvalues"]
target_formula = data["formula_encoding"]
target_smiles = data["smi_encoding"]
target_functionals = data["functional_encoding"]

demo_df = pd.read_pickle("../data/interim/demo-molecule-dataframe.pkl")
results = dict()

# loop through models
for m_index in range(4):
    model = ChainModel.from_paths(
        f"models/Split-SpecEncoder-group{m_index}.pt",
        f"models/Split-FormulaDecoder-group{m_index}.pt",
        f"models/Split-EigenSMILESLSTMDecoder-group{m_index}.pt",
        f"models/Split-ConcatFunctional-group{m_index}.pt",
        cuda="cuda:0"
        # "models/Full-ConcatFunctional.pt"
    )
    model.cuda()
    # loop through examples
    for demo_index, row in tqdm(demo_df.iterrows()):
        name = row["filename"].split("-")[0]
        if name not in results.keys():
            results[name] = dict()
        input_data = constants[demo_index]
        # make sure sign is not used
        print(f"Working on {name}.")
        spectra, formula, smi_coding, functionals = model(input_data, niter=5000, sigma=0.2)
        func_average = functionals.mean(axis=0)
        func_std = np.std(functionals, axis=0) * 2.
        full_smiles = dict()
        smiles_histogram = Counter()
        functional_heatmap = model.functional_group_kde_analysis(functionals)
        # iterate through each predicted encoding, and run beam search
        for iteration in smi_coding:
            decoded = model.beam_search_decoder(iteration, temperature=1.0)
            full_smiles.update(**decoded)
            smiles_histogram += Counter(list(decoded.keys()))
        # store results in nested dictionary
        results[name][m_index] = {
            "decoded_smiles": {
                "predicted": full_smiles,
                "target": target_smiles[demo_index],
                "histogram": smiles_histogram
            },
            "spectra": {
                "predicted": spectra.astype(np.float32),
                "target": target_spectra[demo_index],
            },
            "formula": {
                "predicted": formula.astype(np.float32),
                "target": target_formula[demo_index],
            },
            "functional": {
                "predicted": func_average.astype(np.float32),
                "heatmap": functional_heatmap.astype(np.float32),
                "std": func_std.astype(np.float32),
                "target": target_functionals[demo_index],
            }
        }
dump(results, "demo-results.pkl", compress=9)
