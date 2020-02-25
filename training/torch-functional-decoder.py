from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
import h5py
import wandb

wandb.init(entity="laserkelvin", project="rotconml-functional-decoder")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
print(torch.get_num_threads())
device = "cuda:0"

h5_obj = h5py.File("../../data/processed/newset-processed-split-data.hd5", mode="r")

batch_size = 50
noise_trans = torch_models.WhiteNoise(sigma=10., damping=0.4, shape=(30,))
label_trans = torch_models.LabelSmoothingTransform(weight=0.2)
loss_func = nn.BCELoss(reduction="mean")
tensor_helper = torch_models.ToTorchTensor(device=device)
noise = torch_models.SigmoidWhiteNoise(0.05)

for index in range(4):
    training_dataset = torch_models.DataHandler(
        h5_obj[f"group{index}"]["training"],
        "eigenvalues",
        "functional_encoding",
        [noise_trans, tensor_helper],
        [tensor_helper],
    )

    validation_dataset = torch_models.DataHandler(
        h5_obj[f"group{index}"]["validation"],
        "eigenvalues",
        "functional_encoding",
        [noise_trans, tensor_helper],
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

    opt_settings = {"lr": 7e-4, "weight_decay": 1e-1}

    model = torch_models.FunctionalGroupClassifier(
        opt_settings=opt_settings, loss_func=loss_func, optimizer=torch.optim.AdamW
    )
    model.cuda()
    wandb.watch(model, log="all")

    model.train_model(
        training_loader, epochs=40, validation_dataloader=validation_loader
    )

    torch.save(model.state_dict(), f"models/Split-FunctionalDecoder-group{index}.pt")
    wandb.save(f"models/Split-FunctionalDecoder-group{index}.pt")
    model.dump_history(f"history/Split-FunctionalDecoder-group{index}.yml")
    # make sure we clean up
    del model
