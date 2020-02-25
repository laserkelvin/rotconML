from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import wandb

wandb.init(entity="laserkelvin", project="rotconml-functional-concat")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
print(torch.get_num_threads())
device = "cuda:0"

h5_obj = h5py.File("../../data/processed/newset-processed-split-data.hd5", mode="r")

batch_size = 100
noise_trans = torch_models.WhiteNoise(sigma=20., damping=0.4, shape=(38,))
# label_trans = torch_models.LabelSmoothingTransform(weight=0.2)
loss_func = nn.BCELoss(reduction="mean")
tensor_helper = torch_models.ToTorchTensor(device=device)
noise = torch_models.SigmoidWhiteNoise(0.05)

X_data = np.array(h5_obj["full"]["eigenconcat"])
Y_data = np.array(h5_obj["full"]["functional_encoding"])

train_X, test_X, train_Y, test_Y = train_test_split(X_data, Y_data, test_size=0.3)
training_dataset = torch_models.SimpleDataset(
    train_X, train_Y, [noise_trans, tensor_helper], [noise, tensor_helper]
)
validation_dataset = torch_models.SimpleDataset(
    test_X, test_Y, [noise_trans, tensor_helper], [noise, tensor_helper]
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

opt_settings = {"lr": 1e-4, "weight_decay": 1e-3}

model = torch_models.FunctionalGroupConcat(
    opt_settings=opt_settings, loss_func=loss_func, optimizer=torch.optim.Adam
)
model.cuda()
model.apply(model._initialize_wb)
wandb.watch(model, log="all")

model.train_model(
    training_loader, epochs=40, validation_dataloader=validation_loader
)

torch.save(model.state_dict(), f"models/Full-ConcatFunctional.pt")
wandb.save(f"models/Full-ConcatFunctional.pt")
model.dump_history(f"history/Full-ConcatFunctional.yml")
# make sure we clean up
