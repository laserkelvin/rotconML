
from src.models.torch_models import ChainModel

from joblib import load, dump
import numpy as np
import pandas as pd


# initial settings
npredict = 5000

# load Mike's spreadsheet
df = pd.read_excel("mike-unknowns.xlsx")
# get rid of the NaN for whatever reason
df.dropna(inplace=True)
# Get the unknown molecules
umols = df.loc[df["Name"].str.contains("umol")]

results = dict()
# loop through models and for each model, predict every molecule
for index, row in umols.iterrows():
    inputs = row[["A", "B", "C", "defect", "kappa", "a-type", "b-type", "c-type"]].to_numpy()
    inputs = inputs.astype(np.float32)
    mol_name = row["Name"]
    print(f"Working on {mol_name}.")
    formula_array = np.zeros((4, npredict, 4))
    spectra_array = np.zeros((4, npredict, 30))
    functional_array = np.zeros((4, 23, 100))
    smiles_dict = dict()
    for m_index in range(4):
        model = ChainModel.from_paths(
            f"models/Split-SpecEncoder-group{m_index}.pt",
            f"models/Split-FormulaDecoder-group{m_index}.pt",
            f"models/Split-EigenSMILESLSTMDecoder-group{m_index}.pt",
            f"models/Split-ConcatFunctional-group{m_index}.pt",
        )
        spectra, formula, smi_encoding, functional = model(inputs, npredict)
        for encoding in smi_encoding:
            decoded = model.beam_search_decoder(encoding, temperature=1.)
            smiles_dict.update(**decoded)
        formula_array[m_index] = formula
        spectra_array[m_index] = spectra
        functional_array[m_index] = model.functional_group_kde_analysis(functional)
    results[mol_name] = {
        "formula": formula_array.astype(np.float16),
        "functional": functional_array.astype(np.float16)
    }

dump(results, "mike-identification.pkl")
