from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
import h5py

torch.set_num_threads(8)
torch.set_num_interop_threads(1)

h5_obj = h5py.File("../../data/processed/newset-processed-split-data.hd5", mode="r")

loss_func = nn.MSELoss(reduction="mean")

batch_size = 30
eigen_noise = torch_models.WhiteNoise(20.0, damping=0.4, shape=(30,))
formula_noise = torch_models.PositiveWhiteNoise(0.3)
#minmax_weights = torch_models.MinMaxScaler(max=1.0, min=1.0, target="weight")
#minmax_bias = torch_models.MinMaxScaler(max=1.0, min=1.0, target="bias")

for index in range(4):
    training_dataset = torch_models.DataHandler(
        h5_obj[f"group{index}"]["training"],
        "eigenvalues",
        "formula_encoding",
        [eigen_noise, torch_models.ToTorchTensor()],
        [formula_noise, torch_models.ToTorchTensor()],
    )

    validation_dataset = torch_models.DataHandler(
        h5_obj[f"group{index}"]["validation"], "eigenvalues", "formula_encoding"
    )

    training_loader = data.DataLoader(
        training_dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    validation_loader = data.DataLoader(
        validation_dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    opt_settings = {"lr": 5e-3, "weight_decay": 1.}

    model = torch_models.EigenFormulaDecoder(
        optimizer=torch.optim.AdamW,
        opt_settings=opt_settings,
        loss_func=loss_func,
        batch_norm=True
        #param_transform=[minmax_weights, minmax_bias],
    )

    model.train_model(
        training_loader, epochs=20, validation_dataloader=validation_loader
    )

    torch.save(model, f"models/Split-FormulaDecoderNorm-group{index}.pt")
    # make sure we clean up
    del model
