from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
import h5py
import wandb
import numpy as np
from sklearn.model_selection import train_test_split

wandb.init(entity="laserkelvin", project="rotconml-formula-decoder")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
device = "cuda:0"

h5_obj = h5py.File("../../data/processed/newset-augmented-data.h5", mode="r")

loss_func = nn.L1Loss(reduction="mean")

batch_size = 300
eigen_noise = torch_models.WhiteNoise(10.0, damping=0.4, shape=(38,))
formula_noise = torch_models.PositiveWhiteNoise(0.01)
tensor_helper = torch_models.ToTorchTensor(device=device)
#minmax_weights = torch_models.MinMaxScaler(max=1.0, min=1.0, target="weight")
#minmax_bias = torch_models.MinMaxScaler(max=1.0, min=1.0, target="bias")

for index in range(4):
    X = np.array(h5_obj[f"group{index}"]["eigenconcat"])
    Y = np.array(h5_obj[f"group{index}"]["formula_encoding"])
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    training_dataset = torch_models.SimpleDataset(
        train_X,
        train_Y,
        [tensor_helper],
        [formula_noise, tensor_helper],
    )

    validation_dataset = torch_models.SimpleDataset(
        test_X,
        test_Y,
        [tensor_helper],
        [tensor_helper],
    )

    training_loader = data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    opt_settings = {"lr": 1e-4, "weight_decay": 2e-1}

    model = torch_models.EigenFormulaDecoder(
        optimizer=torch.optim.Adam,
        opt_settings=opt_settings,
        loss_func=loss_func,
        #param_transform=[minmax_weights, minmax_bias],
    )
    model.cuda()
    # model.apply(model._initialize_wb)
    wandb.watch(model, log="all")

    model.train_model(
        training_loader, epochs=80, validation_dataloader=validation_loader
    )

    torch.save(model.state_dict(), f"models/Split-FormulaDecoder-group{index}.pt")
    wandb.save(f"models/Split-FormulaDecoder-group{index}.pt")
    model.dump_history(f"history/Split-FormulaDecoder-group{index}.yml")
    # make sure we clean up
    del model
