
from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
import h5py
import wandb
import numpy as np
from sklearn.model_selection import train_test_split

wandb.init(entity="laserkelvin", project="rotconml-smiles-decoder")

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
print(torch.get_num_threads())
device = "cuda:0"

h5_obj = h5py.File("../../data/processed/newset-augmented-data.h5", mode="r")

batch_size = 500
# noise_trans = torch_models.WhiteNoise(sigma=20., damping=0.4, shape=(100, 4))
label_trans = torch_models.LabelSmoothingTransform(weight=0.1)
loss_func = nn.KLDivLoss(reduction="batchmean")
tensor_helper = torch_models.ToTorchTensor(device=device)

for index in range(4):
    # Get the data
    X = np.array(h5_obj[f"group{index}"]["timeshift_eigenvalues"])
    Y = np.array(h5_obj[f"group{index}"]["smi_encoding"])
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    training_dataset = torch_models.SimpleDataset(
        train_X,
        train_Y,
        [tensor_helper],
        [label_trans, tensor_helper],
    )

    validation_dataset = torch_models.SimpleDataset(
        test_X,
        test_Y,
        [tensor_helper],
        [label_trans, tensor_helper],
    )

    training_loader = data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )

    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
    )

    opt_settings = {"lr": 5e-4, "weight_decay": 1e-2}

    model = torch_models.EigenSMILESLSTMDecoder(
        opt_settings=opt_settings, loss_func=loss_func, optimizer=torch.optim.Adam
    )
    model.cuda()
    model.apply(model._initialize_wb)
    wandb.watch(model, log="all")

    model.train_model(
        training_loader,
        epochs=20,
        validation_dataloader=validation_loader
        )

    torch.save(model.state_dict(), f"models/Split-EigenSMILESLSTMDecoder-group{index}.pt")
    wandb.save(f"models/Split-EigenSMILESLSTMDecoder-group{index}.pt")
    model.dump_history(f"history/Split-EigenSMILESLSTMDecoder-group{index}.yml")
    # make sure we clean up
    del model
