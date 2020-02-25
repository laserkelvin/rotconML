from src.models import torch_models
import torch
from torch import nn
from torch.utils import data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import h5py
import numpy as np
import wandb
import yaml


def calculate_weights(labels_array, factor=2.):
    support = labels_array.sum(axis=0)
    average = np.average(support)
    # find underrepresented labels, and boost them by the inverse
    support_weights = np.ones(support.size) / np.abs(support - average)**(1/factor)
    support_weights /= support_weights.min()
    # Find labels that are actually active
    weights = np.zeros(support.size)
    indices = np.where(support != 0)[0]
    # set the
    weights[indices] = support_weights[indices]
    print(weights)
    return weights


wandb.init(entity="laserkelvin", project="rotconml-functional-concat")

torch.set_num_threads(4)
torch.set_num_interop_threads(1)
print(torch.get_num_threads())
device = "cuda:0"

h5_obj = h5py.File("../../data/processed/newset-augmented-data.h5", mode="r")
demo_obj = h5py.File("../../data/processed/demo-processed-split-data.hd5", "r")["full"]
eigenvalues = demo_obj["eigenvalues"][2]
rotcon = demo_obj["rotcon_kd_dipoles"][2]
demo_data = np.concatenate([eigenvalues, rotcon], axis=-1)
demo_data = torch.from_numpy(demo_data).float().to("cuda:0")

batch_size = 300
noise_trans = torch_models.WhiteNoise(sigma=10., damping=0.4, shape=(38,))
# label_trans = torch_models.LabelSmoothingTransform(weight=0.2)

# loss_func = torch_models.HammingLoss()
tensor_helper = torch_models.ToTorchTensor(device=device)
noise = torch_models.SigmoidWhiteNoise(0.05)
abs_value = torch_models.AbsValue()

for index in range(4):
    # Get the data
    X = np.array(h5_obj[f"group{index}"]["eigenconcat"])
    Y = np.array(h5_obj[f"group{index}"]["functional_encoding"])
    # Calculate which labels are active
    weights = torch.from_numpy(calculate_weights(Y, factor=3.5))
    loss_func = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=weights)
    # split the data
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
    training_dataset = torch_models.SimpleDataset(
        train_X,
        train_Y,
        [tensor_helper],
        [tensor_helper],
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
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

    validation_loader = data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

    opt_settings = {"lr": 1e-5, "weight_decay": 3e-1}

    model = torch_models.FunctionalGroupConv(
        opt_settings=opt_settings, loss_func=loss_func, 
        optimizer=torch.optim.Adam, batch_norm=True
    )
    model.cuda()
    # fan-in fan-out for weights, zero biases
    model.apply(model._initialize_wb)
    wandb.watch(model, log="all")

    model.train_model(
        training_loader, epochs=50, validation_dataloader=validation_loader
    )
    
    # this prevents saving the pos_weights as a parameter
    # which messes up loading the model at inference
    model.loss_func = None
        
    print("-----------------")
    # generate classification report
    temp = torch.from_numpy(train_X).float().cuda(device=device)
    # set model into eval model for the classification report
    model.eval()
    with torch.no_grad():
        Y_pred = model(temp).cpu().numpy()
        Y_pred = np.round(Y_pred).astype(np.uint8)
    report = classification_report(train_Y, Y_pred, output_dict=True)
    with open(f"reports/group{index}-class-report.yml", "w+") as writefile:
        yaml.dump(report, writefile)
    wandb.save(f"reports/group{index}-class-report.yml")
    # Run inference as sanity check
    # with torch.no_grad():
    #     Y_pred = model(demo_data).cpu().numpy()
    # print(Y_pred)
    # Move model back to CPU to see if things change
    model.cpu()
    torch.save(model.state_dict(), f"models/Split-ConcatFunctional-group{index}.pt")
    wandb.save(f"models/Split-ConcatFunctional-group{index}.pt")
    wandb.save(f"torch-concat-functional.py")
    model.dump_history(f"history/Split-ConcatFunctional-group{index}.yml")
    # make sure we clean up
    del model
