
from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
import h5py
import wandb
import numpy as np
from sklearn.model_selection import train_test_split

wandb.init(entity="laserkelvin", project="rotconml-spec-decoder")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
print(torch.get_num_threads())

# device = torch.cuda.device("cuda")
device = "cuda:0"

h5_obj = h5py.File("../../data/processed/newset-augmented-data.h5", mode="r")

loss_func = nn.L1Loss(reduction="mean")

batch_size = 200
# Augment the constants with shifts based on Bayesian model
augmenter = torch_models.AugmentConstants("delta_histograms.csv")
# Add exponentially decaying noise to the Coulomb matrix eigenvalues
noise = torch_models.WhiteNoise(sigma=20., damping=0.4, shape=(30,))
# label_trans = torch_models.LabelSmoothingTransform(sigma=0.1)
tensor_helper = torch_models.ToTorchTensor(device=device)

for index in range(4):
    # Get the data
    X = np.array(h5_obj[f"group{index}"]["rotcon_kd_dipoles"])
    Y = np.array(h5_obj[f"group{index}"]["eigenvalues"])
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    training_dataset = torch_models.SimpleDataset(
        train_X,
        train_Y,
        [augmenter, tensor_helper],
        [tensor_helper],
    )

    validation_dataset = torch_models.SimpleDataset(
        test_X,
        test_Y,
        [augmenter, tensor_helper],
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
    # optimizer = torch_models.Nadam
    opt_settings = {"lr": 7e-4, "weight_decay": 1e-1}

    model = torch_models.EightPickEncoder(
        opt_settings=opt_settings, loss_func=loss_func, optimizer=torch.optim.Adam
    )
    # swap over to CUDA
    model.cuda()
    model.apply(model._initialize_wb)
    wandb.watch(model, log="all")

    model.train_model(
        training_loader,
        epochs=80,
        validation_dataloader=validation_loader
        )

    torch.save(model.state_dict(), f"models/Split-SpecEncoder-group{index}.pt")
    wandb.save(f"models/Split-SpecEncoder-group{index}.pt")
    model.dump_history(f"history/Split-SpecEncoder-group{index}.yml")
    # make sure we clean up
    del model
