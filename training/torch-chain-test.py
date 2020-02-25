
from src.models.torch_models import ChainModel

import numpy as np
from tqdm import tqdm


model = ChainModel.from_paths(
    "models/Split-SpecEncoder-group0.pt",
    "models/Split-FormulaDecoder-group0.pt",
    "models/Split-EigenSMILESLSTMDecoder-group0.pt"
)

test = np.random.normal(size=8)

spectra, formula, smi_coding = model(test, niter=10)

predicted_smiles = dict()

for index, iteration in tqdm(enumerate(smi_coding)):
    smiles = model.beam_search_decoder(iteration, temperature=1.0)
    predicted_smiles.update(**smiles)

predicted_smiles = {k: v for k, v in sorted(predicted_smiles.items(), key=lambda item: item[1])}
print(predicted_smiles)
