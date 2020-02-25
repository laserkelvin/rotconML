from typing import Tuple
import re
from collections import Counter

from src.pipeline.parse_calculations import timeshift_array

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
import numba
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data
from torch import nn, optim
from torch.nn import functional as F


def smi_sanitizer(smi: str) -> str:
    """
    A set of logical statements that try and clean up
    predicted SMILES codes so that they're not garbage.
    
    Parameters
    ----------
    smi : str
        [description]
    
    Returns
    -------
    str
        [description]
    """
    # Loop through each character of the SMILES
    smi = smi.replace(" ", "")
    # remove chirality
    smi = smi.replace("@", "")
    # remove explicit hydrogen
    smi = smi.replace("H", "")
    # Look for brackets that hang and delete them
    pair_targets = ["[]", "()"]
    for pair in pair_targets:
        smi = smi.replace(pair, "")
    smi = re.sub(r"\W{2,3}", "", smi)
    smi = re.sub(r"(\W\d|\d\W)", "", smi)
    # Loop through and ensure values that come in two
    # actually do so
    char_count = Counter(smi)
    # This makes sure that the ring is closed properly
    pair_targets = [str(i) for i in range(1, 9)]
    for symbol, n_symbol in char_count.items():
        if (symbol in pair_targets) and (n_symbol % 2 != 0):
            smi = smi.replace(symbol, "", 1)
    # Try and balance the brackets
    smi = bracket_balancer(smi)
    try:
        # SMILES codes can't begin with non-alphabet
        if not smi[0].isalpha():
            smi = smi[1:]
        while smi[-1] in ["#", "=", "+", "-"]:
            smi = smi[:-1]
        return smi
    except IndexError:
        print(smi)
        return "blank"


def bracket_balancer(smi: str) -> str:
    """
    Function that attempts to balance brackets. The logic
    is not currently perfect as there is still a possibility
    for hanging brackets, albeit this significantly cleans
    most of them up.
    
    Parameters
    ----------
    smi : str
        Input SMILES string
    
    Returns
    -------
    str
        Cleaned SMILES string
    """
    temp = [char for char in smi]
    queue = list()
    for index, char in enumerate(temp):
        if char in ["(", "["]:
            queue.append(char)
            continue
        # if a closing bracket is found, check if its counterpart is in
        # the queue; if not, it means this is a hanging. If it does exist,
        # remove it from the queue.
        if char == ")":
            if "(" in queue:
                queue.remove("(")
            else:
                temp.pop(index)
            continue
        if char == "]":
            if "[" in queue:
                queue.remove("[")
            else:
                temp.pop(index)
            continue
    # Clear remaining hanging brackets
    for char in queue:
        temp.remove(char)
    return "".join(temp)


class AugmentConstants(object):
    """
    Class for transforming rotational constants. This class
    is first initialized by providing values of delta and
    associated probability values. Each time this class is
    called, it will randomly sample from delta, and shift
    the training data by the sampled delta.
    
    Parameters
    ----------
    object : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """

    def __init__(self, p_dist_csv):
        self.dist_df = pd.read_csv(p_dist_csv)
        self.delta = self.dist_df["Bins"].values
        self.p_delta = self.dist_df["Modeled"].values

    def __call__(self, sample: np.ndarray):
        delta = np.random.choice(self.delta, size=3, replace=True, p=self.p_delta)
        shift = 100.0 - delta
        sample[:3] = (100.0 * sample[:3]) / shift
        return sample


class AbsValue(object):
    def __call__(self, sample: np.ndarray):
        return np.abs(sample)


class WhiteNoise(object):
    """
    Transformation for augmenting samples with
    Gaussian noise of a specified width. To be used
    in conjunction with torch transforms.
    
    The `damping` argument provides a way to add an exponential
    decay to the noise - since for Coulomb matrix eigenvalues
    the leading values are much larger than the tail.

    Returns
    -------
    [type]
        [description]
    """

    def __init__(self, sigma=20.0, damping=0.4, shape=(30,)):
        self.damping = 1.0
        self.sigma = sigma
        if damping:
            seq_length = shape[-1]
            x = np.linspace(1.0, 5.0, seq_length)
            self.damping = np.exp(-x * damping)
            if len(shape) > 1:
                self.damping = timeshift_array(self.damping, seq_length, shape[0])

    def __call__(self, sample: np.ndarray):
        noise = np.random.normal(scale=self.sigma, size=sample.shape)
        return sample + (noise * self.damping)


class PositiveWhiteNoise(object):
    """
    Transformation that adds Gaussian noise to the sample,
    and ensures that the resulting values are always positive.
    """

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, sample: np.ndarray):
        noise = np.random.normal(scale=self.sigma, size=sample.shape)
        new_sample = noise + sample
        new_sample[new_sample < 0.8] = 0.0
        return new_sample


class HammingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y):
        mislabel = torch.sum(y_pred != y).float()
        return mislabel / y.shape[-1]


class SigmoidWhiteNoise(PositiveWhiteNoise):
    def __init__(self, sigma=0.5):
        super().__init__(sigma)

    def __call__(self, sample: np.ndarray):
        noise = np.random.normal(scale=self.sigma, size=sample.shape)
        new_sample = noise + sample
        new_sample[new_sample < 0.0] = 0.0
        new_sample[new_sample > 1.0] = 1.0
        return new_sample


class OODInputScrambling(object):
    """
    Transformation that will scramble an input array.
    This is used for the out-of-distribution OOD model
    training. The likelihood p determines the frequency
    that the arrays are completely scrambled.
    
    This is as described in arXiv:1906.02845v2 [stat.ML],
    where they really only describe it for discrete data
    and image pixels.
    
    In this application, probability `p` determines what
    the likelihood of generating "noise data" based on the
    samples is. This is generated by uniform sampling using
    0 and the maximum value in `sample`
    """

    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, sample: np.ndarray):
        scramble = np.random.binomial(1, self.p)
        if scramble == 1:
            new_data = np.random.uniform(low=0.0, high=sample.max(), size=sample.size)
            return new_data
        else:
            return sample


class LabelSmoothingTransform(object):
    """
    Rudimentary version of label smoothing as described here:
    https://leimao.github.io/blog/Label-Smoothing/, which is
    subsequently based off this arxiv paper:
    https://arxiv.org/pdf/1512.00567.pdf
    
    The idea behind this method is to improve regularization,
    or decrease model confidence. 
    """

    def __init__(self, weight=0.1):
        self.weight = weight

    def __call__(self, sample: torch.Tensor):
        # Uniform noise based on number of classes K
        noise = torch.ones(sample.shape) / sample.shape[-1]
        # Multiply noise by the weights
        smoothed_labels = noise * self.weight
        # Add the original one-hot encoded values back
        smoothed_labels[sample != 0] += 1.0 - self.weight
        # Renormalize likelihoods
        smoothed_labels = smoothed_labels / smoothed_labels.sum(axis=1)[:, None]
        return smoothed_labels.numpy()


class ToTorchTensor(object):
    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample: np.ndarray):
        return torch.from_numpy(sample)  # .to(device=self.device)


class MinMaxScaler(object):
    """
    Transformation that will scale values within a specified range.
    This can be used to constrain parameters to within a certain range,
    as a form of regularization.
    """

    def __init__(self, max=1.0, min=-1.0, frequency=5.0, target="weight"):
        self.max = 1.0
        self.min = 1.0
        self.target = target

    def __call__(self, module: torch.nn.Module):
        if hasattr(module, self.target):
            target = getattr(module, self.target)
            values = target.data
            tar_max = values.max(dim=-1)[0].unsqueeze(-1)
            tar_min = values.min(dim=-1)[0].unsqueeze(-1)
            values.sub_(tar_min)
            values.div_(tar_max - tar_min)
            values *= (self.max - self.min) + self.min


class DataHandler(data.Dataset):
    def __init__(self, h5_link, X_key, Y_key, x_transform=None, y_transform=None):
        """
        Specialized DataHandler class that inherits from the PyTorch
        `Dataset` base class. Once initialized, this object can be passed
        to a torch `DataLoader`, which will then handle the data streaming.
        
        Parameters
        ----------
        h5_link : [type]
            [description]
        X_key, Y_key : [type]
            [description]
        transform : list or Compose object, optional
            A chain of transformations to be applied to the data when it is
            loaded, by default None. The transformations can be provided
            as a list, or as a torchvision Compose object.
        """
        super().__init__()
        # set up the links to the HDF5 data, but don't
        # actually load anything yet
        self.X = h5_link[X_key]
        self.Y = h5_link[Y_key]
        # If the transform argument is a list, convert it into
        # a torchvision Compose
        if type(x_transform) == list:
            x_transform = transforms.Compose(x_transform)
        self.x_transform = x_transform
        if type(y_transform) == list:
            y_transform = transforms.Compose(y_transform)
        self.y_transform = y_transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        # get data from the HDF5 file, load into memory
        x = self.X[index]
        y = self.Y[index]
        # if there are some transformations specified,
        # do it before we return the data
        if self.x_transform:
            x = self.x_transform(x)
        else:
            x = torch.from_numpy(x)
        if self.y_transform:
            y = self.y_transform(y)
        else:
            y = torch.from_numpy(y)
        return (x.float(), y.float())


class SimpleDataset(DataHandler):
    def __init__(self, x_data, y_data, x_transform=None, y_transform=None):
        super(data.Dataset).__init__()
        self.X = x_data
        self.Y = y_data
        if type(x_transform) == list:
            x_transform = transforms.Compose(x_transform)
        self.x_transform = x_transform
        if type(y_transform) == list:
            y_transform = transforms.Compose(y_transform)
        self.y_transform = y_transform


class Nadam(optim.Optimizer):
    """Implements Nadam algorithm (a variant of Adam based on Nesterov momentum).

    It has been proposed in `Incorporating Nesterov Momentum into Adam`__.

    :param params: (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
    :param lr: (float, optional): learning rate (default: 2e-3)
    :param betas: (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square
    :param eps: (float, optional): term added to the denominator to improve
        numerical stability (default: 1e-8)
    :param weight_decay: (float, optional): weight decay (L2 penalty) (default: 0)
    :param schedule_decay: (float, optional): momentum schedule decay (default: 4e-3)

    __ http://cs229.stanford.edu/proj2015/054_report.pdf
    __ http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

    """

    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        schedule_decay=4e-3,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            schedule_decay=schedule_decay,
        )
        super(Nadam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        :param closure: (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m_schedule"] = 1.0
                    state["exp_avg"] = grad.new().resize_as_(grad).zero_()
                    state["exp_avg_sq"] = grad.new().resize_as_(grad).zero_()

                # Warming momentum schedule
                m_schedule = state["m_schedule"]
                schedule_decay = group["schedule_decay"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(group["weight_decay"], p.data)

                momentum_cache_t = beta1 * (
                    1.0 - 0.5 * (0.96 ** (state["step"] * schedule_decay))
                )
                momentum_cache_t_1 = beta1 * (
                    1.0 - 0.5 * (0.96 ** ((state["step"] + 1) * schedule_decay))
                )
                m_schedule_new = m_schedule * momentum_cache_t
                m_schedule_next = m_schedule * momentum_cache_t * momentum_cache_t_1
                state["m_schedule"] = m_schedule_new

                # Decay the first and second moment running average coefficient
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq_prime = exp_avg_sq.div(1.0 - bias_correction2)

                denom = exp_avg_sq_prime.sqrt_().add_(group["eps"])

                p.data.addcdiv_(
                    -group["lr"] * (1.0 - momentum_cache_t) / (1.0 - m_schedule_new),
                    grad,
                    denom,
                )
                p.data.addcdiv_(
                    -group["lr"] * momentum_cache_t_1 / (1.0 - m_schedule_next),
                    exp_avg,
                    denom,
                )

        return loss


class GenericModel(nn.Module):
    def __init__(self, loss_func=None, param_transform=None, tracker=True):
        super().__init__()
        self.optimizer = None
        # Define some default values for loss functions.
        self.loss_func = loss_func
        if param_transform:
            if type(param_transform) == list:
                self.param_transform = transforms.Compose(param_transform)
        self.param_transform = param_transform
        self.grad_history = list()
        self.epoch_grad_history = list()
        self.epoch_param_history = list()
        self.loss_history = list()
        self.validation_history = list()
        self.tracker = tracker

    @classmethod
    def load_weights(cls, weights_path: str, device=None, batch_norm=False):
        """
        Convenience method for loading in the weights of a model.
        Basically initializes the model, and wraps a `torch.load`
        with automatic cuda/cpu detection.
        
        Parameters
        ----------
        weights_path : str
            String path to the trained weights of a model; typically
            with extension .pt
        
        Returns
        -------
        model
            Instance of the PyTorch model with loaded weights
        """
        if not device:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        model = cls(batch_norm=batch_norm)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        return model

    def init_layers(self, weight_func=None, bias_func=None):
        """
        Function that will initialize all the weights and biases of
        the model layers. This function uses the `apply` method of
        `Module`, and so will only work on layers that are contained
        as children.
        
        Parameters
        ----------
        weight_func : `nn.init` function, optional
            Function to use to initialize weights, by default None
            which will default to `nn.init.xavier_normal`
        bias_func : `nn.init` function, optional
            Function to use to initialize biases, by default None
            which will default to `nn.init.xavier_uniform`
        """
        if not weight_func:
            self._weight_func = nn.init.xavier_normal
        if not bias_func:
            self._bias_func = nn.init.xavier_uniform
        # Apply initializers to all of the Module's children with `apply`
        self.apply(self._initialize_wb)

    def _initialize_wb(self, layer: nn.Module):
        """
        Static method for applying an initializer to weights
        and biases. If a layer is passed without weight and
        bias attributes, this function will effectively ignore it.
        
        Parameters
        ----------
        layer : `nn.Module`
            Layer that is a subclass of `nn.Module`
        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias.data)
        elif isinstance(layer, nn.LSTM):
            torch.nn.init.xavier_uniform_(layer.weight_hh_l0)
            torch.nn.init.xavier_uniform_(layer.weight_ih_l0)
            if layer.bias_ih_l0 is not None:
                torch.nn.init.zeros_(layer.bias_ih_l0)
            if layer.bias_hh_l0 is not None:
                torch.nn.init.zeros_(layer.weight_hh_l0)
        if isinstance(layer, nn.BatchNorm1d):
            torch.nn.init.zeros_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.ones_(layer.bias.data)

    def __len__(self):
        return sum([param.numel() for param in self.parameters()])

    def dump_history(self, filepath: str):
        """
        Method for saving the training history into YAML format for
        easy read/write/plotting.
        
        Parameters
        ----------
        filepath : str
            Filepath to save dump to. Does not include extension .yml.
        """
        data_dict = {
            "epochs": self.epoch_list,
            # "gradients": self.grad_history.tolist(),
            "training_loss": [float(value) for value in self.loss_history],
            "validation_loss": [float(value) for value in self.validation_history],
        }
        with open(filepath, "w+") as write_file:
            yaml.dump(data_dict, write_file)

    def train_model(
        self,
        dataloader: "DataLoader",
        epochs=10,
        progress=True,
        validation_dataloader=None,
        cuda=True,
    ):
        """
        Generalized function for performing model training. All this function
        does is automate some of the common procedures, i.e. keeping track of
        training and validation statistics over minibatch and epochs.
        
        The procedure is the simplest as can be: for each epoch, iterate
        through the dataset, calculating the model prediction for each sample
        in the training set, calculate the loss, backpropagate gradients and
        then update the parameters accordingly.
        
        The results are saved as model attributes, e.g. model.grad_history
        for later inspection.
        
        Parameters
        ----------
        dataloader : `torch.utils.DataLoader`
            PyTorch dataloader instance. Training loop will iterate through
            this set via a for loop.
        epochs : int, optional
            [description], by default 10
        progress : bool, optional
            [description], by default True
        validation_dataloader : [type], optional
            [description], by default None
        """
        self.epoch_list = list()
        for epoch in tqdm(range(epochs)):
            iteration_loss = list()
            self.train()
            for x, y in dataloader:
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                mean_loss = self._train_iteration(x, y, train=True)
                # Calculate the norm of gradients for every parameter
                # for storing in the training history
                gradients = [
                    param.grad.norm().cpu().numpy() for param in self.parameters()
                ]
                self.grad_history.append(gradients)
                iteration_loss.append(mean_loss)
            training_avg_loss = np.mean(iteration_loss)
            self.loss_history.append(training_avg_loss)
            print(f"Training loss for epoch {epoch}:\t{training_avg_loss:.2f}")
            # Every epoch we'll take a checkpoint of the gradients and parameters
            self.epoch_grad_history.append(
                [param.grad.cpu().numpy() for param in self.parameters()]
            )
            # If validation data is provided, see how the the model predicts the
            # validation set
            if validation_dataloader:
                epoch_val = list()
                self.eval()
                for val_x, val_y in validation_dataloader:
                    if cuda:
                        val_x = val_x.cuda()
                        val_y = val_y.cuda()
                    val_mean_loss = self._train_iteration(val_x, val_y, train=False)
                    epoch_val.append(val_mean_loss)
                val_avg_loss = np.mean(epoch_val)
                self.validation_history.append(val_avg_loss)
                print(f"Validation loss for epoch {epoch}:\t{val_avg_loss:.2f}")
            if self.tracker:
                wandb.log(
                    {
                        "Training Loss": training_avg_loss,
                        "Validation Loss": val_avg_loss,
                    }
                )
            self.epoch_list.append(epoch + 1)

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, train=True
    ) -> torch.Tensor:
        """
        Calculate the loss for a set of inputs and the ground truth.
        The `train` argument specifies whether or not gradients are needed;
        during validation we want to turn off grad tracking.
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        y : torch.Tensor
            [description]
        train : bool, optional
            [description], by default True
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if train:
            pred_y = self.forward(x)
            loss = self.loss_func(pred_y, y)
        else:
            with torch.no_grad():
                pred_y = self.forward(x)
                loss = self.loss_func(pred_y, y)
        return loss

    def _train_iteration(
        self, x: torch.Tensor, y: torch.Tensor, train=True
    ) -> torch.Tensor:
        """
        Private method for 
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        y : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        loss = self.compute_loss(x, y, train=train)
        # In training mode, backpropagate the gradients
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     self.parameters(), 1.
            # )
            self.optimizer.step()
            # If we are to impose some regularization, do it
            # after the optimization step
            if self.param_transform:
                self.apply(self.param_transform)
        return loss.mean().item()

    def get_num_parameters(self) -> int:
        """
        Calculate the number of parameters contained within the model.
        
        Returns
        -------
        int
            Number of trainable parameters
        """
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class EightPickEncoder(GenericModel):
    """
    Encoder model that will convert the spectroscopic parameters
    into coulomb matrix eigenvalues.
    
    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        tracker=True,
    ):
        super().__init__(
            loss_func=loss_func, param_transform=param_transform, tracker=tracker
        )
        self.dense_1 = nn.Linear(in_features=8, out_features=32)
        self.dense_2 = nn.Linear(in_features=32, out_features=64)
        self.dense_3 = nn.Linear(in_features=64, out_features=128)
        self.dense_4 = nn.Linear(in_features=128, out_features=64)
        self.dense_5 = nn.Linear(in_features=64, out_features=30)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**optimizer_dict)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the network using a specified
        set of inputs. The dropout layers are included in the
        `forward` 
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        if x.dtype != torch.float32:
            x = x.float()
        output = F.leaky_relu(self.dense_1(x), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_2(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_3(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_4(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.relu(self.dense_5(output))
        return output


class EigenSMILESLSTMDecoder(GenericModel):
    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        tracker=True,
    ):
        super().__init__(
            loss_func=loss_func, param_transform=param_transform, tracker=tracker
        )
        # self.batch_norm = nn.BatchNorm1d(4)
        # Use a three-stack, bidirectional LSTM layer with dropouts
        self.lstm_1 = nn.LSTM(4, 100, num_layers=4, batch_first=True, dropout=0.3)
        self.lstm_2 = nn.LSTM(100, 200, num_layers=3, batch_first=True, dropout=0.4)
        self.lstm_3 = nn.LSTM(200, 400, num_layers=3, batch_first=True, dropout=0.4)
        self.linear_1 = nn.Linear(400, 200)
        self.linear_2 = nn.Linear(200, 30, bias=False)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**opt_settings)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        # Perform normalization across values
        # output = self.batch_norm(x.permute(0, 2, 1))
        # First LSTM pass, after restoring original permutation
        # output, (h, c) = self.lstm_1(output.permute(0, 2, 1))
        output, (h, c) = self.lstm_1(x)
        output = F.leaky_relu(output, 0.5)
        output, (h, c) = self.lstm_2(output)
        output = F.leaky_relu(output, 0.5)
        output, (h, c) = self.lstm_3(output)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.linear_1(output), 1.0)
        # Perform dropouts after first linear layer
        output = F.dropout(output, p=0.3)
        # Final layer with softmax activation, where dim specifies
        # that probabilities along each "character" adds up to 1
        output = self.linear_2(output)
        return output

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, train=True
    ) -> torch.Tensor:
        """
        Calculate the loss for a set of inputs and the ground truth.
        The `train` argument specifies whether or not gradients are needed;
        during validation we want to turn off grad tracking.
        
        This version of `compute_loss` is slightly different because
        we're going to use KL-divergence as the metric; as implemented in
        PyTorch, the predictions have to be log likelihoods and so we
        apply a `log_softmax` activation prior to calculating the loss.
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        y : torch.Tensor
            [description]
        train : bool, optional
            [description], by default True
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if train:
            pred_y = F.log_softmax(self.forward(x), dim=2)
            loss = self.loss_func(pred_y, y)
        else:
            with torch.no_grad():
                pred_y = F.log_softmax(self.forward(x), dim=2)
                loss = self.loss_func(pred_y, y)
        return loss


class EigenFormulaDecoder(GenericModel):
    """
    Decoder model that will convert Coulomb matrix eigenvalues
    into chemical formula.
    
    Parameters
    ----------
    nn : [type]
        [description]
    """

    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        batch_norm=False,
        tracker=True,
    ):
        super().__init__(
            loss_func=loss_func, param_transform=param_transform, tracker=tracker
        )
        if batch_norm:
            self.normalize = True
            self.batch_norm = nn.BatchNorm1d(38)
        else:
            self.normalize = False
        self.dense_1 = nn.Linear(in_features=38, out_features=128)
        self.dense_2 = nn.Linear(in_features=128, out_features=256)
        self.dense_3 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.dense_4 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.dense_5 = nn.Linear(in_features=64, out_features=32, bias=True)
        # Last layer does not use bias, as it tends to swamp the result
        self.dense_6 = nn.Linear(in_features=32, out_features=4, bias=True)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**optimizer_dict)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)
        # self.apply(self.init_biases)

    @staticmethod
    def init_biases(module):
        bias_layer = getattr(module, "bias", None)
        if bias_layer is not None:
            bias_layer.data.fill_(1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the network using a specified
        set of inputs. The dropout layers are included in the
        `forward`
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        if x.dtype != torch.float32:
            x = x.float()
        # If we want to start doing batch normalization. The `getattr`
        # ensures backwards compatibility
        if getattr(self, "normalize", False):
            output = self.batch_norm(x)
        else:
            output = x
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_1(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_2(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_3(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_4(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_5(output), 0.3)
        output = F.dropout(output, p=0.3)
        # Final output layer with ReLU activation to make
        # the atom numbers always positive
        output = F.relu(self.dense_6(output))
        return output


class FunctionalGroupClassifier(GenericModel):
    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        batch_norm=False,
        tracker=True,
    ):
        super().__init__(
            loss_func=loss_func, param_transform=param_transform, tracker=tracker
        )
        if batch_norm:
            self.normalize = True
            self.batch_norm = nn.BatchNorm1d(30)
        else:
            self.normalize = False
        self.dense_1 = nn.Linear(in_features=30, out_features=256)
        self.dense_2 = nn.Linear(in_features=256, out_features=512)
        self.dense_3 = nn.Linear(in_features=512, out_features=256,)
        self.dense_4 = nn.Linear(in_features=256, out_features=64)
        self.dense_5 = nn.Linear(in_features=64, out_features=32)
        # Last layer does not use bias, as it tends to swamp the result
        self.dense_6 = nn.Linear(in_features=32, out_features=23)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**optimizer_dict)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)
        # self.apply(self.init_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the network using a specified
        set of inputs. The dropout layers are included in the
        `forward`
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        if x.dtype != torch.float32:
            x = x.float()
        # If we want to start doing batch normalization. The `getattr`
        # ensures backwards compatibility
        if getattr(self, "normalize", False):
            output = self.batch_norm(x)
        else:
            output = x
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_1(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_2(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_3(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_4(output), 0.3)
        output = F.dropout(output, p=0.3)
        output = F.leaky_relu(self.dense_5(output), 0.3)
        output = F.dropout(output, p=0.3)
        # Final layer uses sigmoid activation
        output = torch.sigmoid(self.dense_6(output))
        return output


class FunctionalGroupConcat(GenericModel):
    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        batch_norm=False,
        tracker=True,
        inference=True
    ):
        super().__init__(
            loss_func=loss_func, param_transform=param_transform, tracker=tracker
        )
        if batch_norm:
            self.normalize = True
            self.batch_norm = nn.BatchNorm1d(38)
            if inference:
                self.batch_norm.eval()
        else:
            self.normalize = False
        self.dense_1 = nn.Linear(in_features=38, out_features=256)
        self.dense_2 = nn.Linear(in_features=256, out_features=512)
        self.dense_3 = nn.Linear(in_features=512, out_features=986)
        self.dense_4 = nn.Linear(in_features=1024, out_features=512)
        self.dense_5 = nn.Linear(in_features=512, out_features=256)
        self.dense_6 = nn.Linear(in_features=256, out_features=128)
        self.dense_7 = nn.Linear(in_features=128, out_features=128)
        self.dense_8 = nn.Linear(in_features=128, out_features=64)
        self.dense_9 = nn.Linear(in_features=64, out_features=23)
        self.prelu_1 = nn.PReLU(1)
        self.prelu_2 = nn.PReLU(1)
        self.prelu_3 = nn.PReLU(1)
        self.prelu_4 = nn.PReLU(1)
        self.prelu_5 = nn.PReLU(1)
        self.prelu_6 = nn.PReLU(1)
        self.prelu_7 = nn.PReLU(1)
        self.prelu_8 = nn.PReLU(1)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**optimizer_dict)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)
        # self.apply(self.init_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the network using a specified
        set of inputs. The dropout layers are included in the
        `forward`
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        if x.dtype != torch.float32:
            x = x.float()
        # If we want to start doing batch normalization. The `getattr`
        # ensures backwards compatibility
        if getattr(self, "normalize", False):
            output = self.batch_norm(x)
        else:
            output = x
        # output = F.dropout(output, p=0.3)
        output = self.prelu_1(self.dense_1(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_2(self.dense_2(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_3(self.dense_3(output))
        output = F.dropout(output, p=0.2)
        # Skip layer, includes the inputs
        output = torch.cat([output, x], dim=-1)
        output = self.prelu_4(self.dense_4(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_5(self.dense_5(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_6(self.dense_6(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_7(self.dense_7(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_8(self.dense_8(output))
        output = F.dropout(output, p=0.2)
        output = self.dense_9(output)
        # Final layer uses sigmoid activation if using binary cross-entropy
        if (
            isinstance(self.loss_func, (nn.BCELoss, nn.MSELoss))
            or self.loss_func is None
        ):
            output = torch.sigmoid(output)
        # do nothing for loss functions that include sigmoid
        else:
            pass
        return output


class FunctionalGroupConv(GenericModel):
    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        batch_norm=False,
        tracker=True,
        inference=False
    ):
        super().__init__(
            loss_func=loss_func, param_transform=param_transform, tracker=tracker
        )
        if batch_norm:
            self.normalize = True
            self.batch_norm = nn.BatchNorm1d(38)
            if inference:
                self.batch_norm.eval()
        else:
            self.normalize = False
        # input layer
        self.dense_1 = nn.Linear(in_features=38, out_features=256)
        self.dense_2 = nn.Linear(in_features=256, out_features=512)
        self.conv_1 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.LeakyReLU(0.3, inplace=True),
            nn.MaxPool1d(1),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(128, 512, kernel_size=3),
            nn.LeakyReLU(0.3, inplace=True),
            nn.MaxPool1d(2),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.MaxPool1d(1),
        )
        self.dense_3 = nn.Linear(in_features=256, out_features=128)
        self.dense_4 = nn.Linear(in_features=128, out_features=64)
        self.dense_5 = nn.Linear(in_features=64, out_features=23)
        self.prelu_1 = nn.PReLU(1)
        self.prelu_2 = nn.PReLU(1)
        self.prelu_3 = nn.PReLU(1)
        self.prelu_4 = nn.PReLU(1)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**optimizer_dict)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)
        # self.apply(self.init_biases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the network using a specified
        set of inputs. The dropout layers are included in the
        `forward`
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        if x.dtype != torch.float32:
            x = x.float()
        # If we want to start doing batch normalization. The `getattr`
        # ensures backwards compatibility
        if getattr(self, "normalize", False):
            output = self.batch_norm(x)
        else:
            output = x
        # output = F.dropout(output, p=0.3)
        output = self.prelu_1(self.dense_1(output))
        output = F.dropout(output, p=0.2)
        output = self.prelu_2(self.dense_2(output))
        output = F.dropout(output, p=0.2)
        output = self.conv_1(output.view(output.shape[0], 64, 8))
        output = self.conv_2(output)
        output = self.conv_3(output)
        # Reshape back into linear layers
        batchsize, nsequence, seq_length = output.shape
        output = F.dropout(output.view(output.shape[0], nsequence * seq_length), p=0.2)
        output = self.prelu_3(self.dense_3(output))
        output = F.dropout(output, p=0.2)
        # Skip layer, includes the inputs
        # output = torch.cat([output, x], dim=-1)
        output = self.prelu_4(self.dense_4(output))
        output = F.dropout(output, p=0.2)
        output = self.dense_5(output)
        # Final layer uses sigmoid activation if using binary cross-entropy
        if (
            isinstance(self.loss_func, (nn.BCELoss, nn.MSELoss))
            or self.loss_func is None
        ):
            output = torch.sigmoid(output)
        # do nothing for loss functions that include sigmoid
        else:
            pass
        return output


class CoulombMatrixRebuilder(GenericModel):
    def __init__(
        self,
        optimizer=None,
        loss_func=None,
        opt_settings=None,
        param_transform=None,
        batch_norm=False,
    ):
        super().__init__(loss_func=loss_func, param_transform=param_transform)
        self.conv_1 = nn.Sequential(
            nn.Conv1d(100, 200, 2, padding=2), nn.LeakyReLU(0.1), nn.MaxPool1d(2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(200, 400, 2, padding=2), nn.LeakyReLU(0.1), nn.MaxPool1d(2)
        )
        # This linear layer is basically here only to get the
        # array into the right shape
        self.linear_1 = nn.Linear(1200, 900)
        if not optimizer:
            optimizer = optim.Adam
        optimizer_dict = {"lr": 1e-3}
        if opt_settings:
            optimizer_dict.update(**optimizer_dict)
        self.optimizer = optimizer(self.parameters(), **optimizer_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass of the network using a specified
        set of inputs. The dropout layers are included in the
        `forward`
        
        Parameters
        ----------
        x : torch.Tensor
            [description]
        
        Returns
        -------
        torch.Tensor
            [description]
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).float()
        if x.dtype != torch.float32:
            x = x.float()
        output = self.conv_1(x)
        output = self.conv_2(output)
        batch_size = output.shape[0]
        # This flattens the array to be fed into a linear layer
        output = output.view(batch_size, -1)
        output = F.relu(self.linear_1(output))
        return output.view(batch_size, 30, 30)


class ChainModel(nn.Module):
    def __init__(
        self,
        spec_encoder,
        formula_decoder,
        smiles_decoder,
        functional_decoder,
        cuda=None,
    ):
        super().__init__()
        self.spec_encoder = spec_encoder
        self.formula_decoder = formula_decoder
        self.smiles_decoder = smiles_decoder
        self.functional_decoder = functional_decoder
        self.name = "MoleculeDetective"
        # leave decision up to machinery if cuda not specified
        if cuda is None:
            self._cuda = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._cuda = cuda
        # If cuda enabled, move model to GPU
        if self._cuda == "cuda":
            self.cuda()

    def forward(
        self, X: np.ndarray, niter=1000, deterministic=False, sigma=0.0, gradients=False
    ):
        if deterministic:
            self.train = True
        else:
            self.train = False
        if X.ndim != 1 or X.size != 8:
            raise ValueError(
                "Dimensionality or length of X is incorrect; "
                "please provide a 1D array that is 8 elements long."
            )
        # Repeat inputs niter times to vectorize inference
        X_input = np.tile(X, (niter, 1))
        # If performing bootstrap, add Gaussian noise to inputs
        if sigma > 0:
            noise = np.random.normal(scale=sigma, size=(niter - 1, 8))
            X_input[1:] += noise
        X_input = torch.from_numpy(X_input).float()
        if self._cuda:
            X_input = X_input.cuda()
        if not gradients:
            inference_func = self._nograd_inference_chain
        else:
            inference_func = self._inference_chain
        # Run the inference
        spectra, formula, smiles_encoding, functional_groups = inference_func(X_input)
        # Explicitly move everything back to CPU
        self.spectra = spectra.cpu()
        self.formula = formula.cpu()
        self.smiles_encoding = smiles_encoding.cpu()
        self.functional_groups = functional_groups.cpu()
        return (
            self.spectra.numpy(),
            self.formula.numpy(),
            self.smiles_encoding.numpy(),
            self.functional_groups.numpy(),
        )

    def _nograd_inference_chain(self, X: torch.Tensor):
        with torch.no_grad():
            (
                spectra,
                formula,
                smiles_encoding,
                functional_groups,
            ) = self._inference_chain(X)
        return spectra, formula, smiles_encoding, functional_groups

    def _inference_chain(self, X: np.ndarray):
        spectra = self.spec_encoder(X)
        concat_eigen = torch.cat([spectra, X], dim=-1)
        formula = self.formula_decoder(concat_eigen)
        # Create windows of the predicted eigenspectra
        shifted_spectra = np.apply_along_axis(
            timeshift_array, -1, spectra.cpu().numpy()
        )
        shifted_spectra = torch.from_numpy(shifted_spectra)
        if self._cuda:
            shifted_spectra = shifted_spectra.cuda()
        smiles_encoding = F.softmax(self.smiles_decoder(shifted_spectra), dim=-1)
        smiles_encoding.squeeze_(1)
        # Calculate formula likelihoods
        functional_groups = self.functional_decoder(concat_eigen)
        return spectra, formula, smiles_encoding, functional_groups

    @staticmethod
    def functional_group_kde_analysis(
        functional_array: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Function that will use Kernel Density estimation to "histogram" the
        probability distribution of functional groups. The result is a 2D array
        that can be used to make a heatmap, where the first dimension corresponds
        to the functional group index, and the second dimension the likelihood
        of that functional group being present.
        
        Parameters
        ----------
        functional_array : np.ndarray
            [description]
        
        Returns
        -------
        np.ndarray
            NumPy 2D array corresponding to the functional group heatmap
        """
        default_kde = {"bandwidth": 0.03, "kernel": "tophat"}
        default_kde.update(**kwargs)
        niterations, nlabels = functional_array.shape
        x_values = np.linspace(0.0, 1.0, 100)
        kde_heatmap = np.zeros((nlabels, x_values.size))
        for label_index in range(nlabels):
            kde = KernelDensity(**default_kde)
            kde.fit(functional_array[:, label_index][:, None])
            heatmap = np.exp(kde.score_samples(x_values[:, None]))
            # normalize likelihoods in each label
            kde_heatmap[label_index] = heatmap / heatmap.sum()
        return kde_heatmap

    @staticmethod
    @numba.njit(fastmath=True)
    def _thermalize_likelihoods(X: np.ndarray, T: float, log=False) -> np.ndarray:
        """
        Re-calculate likelihoods 
        
        Parameters
        ----------
        X : np.ndarray
            [description]
        T : float
            [description]
        
        Returns
        -------
        np.ndarray
            [description]
        """
        logp = np.log(X) / T
        newp = np.exp(logp)
        p = np.zeros(X.shape, dtype=np.float64)
        # If the array is 2D, make sure normalization is performed
        # along the rows
        if X.ndim == 2:
            p = newp / newp.sum(axis=-1).reshape(-1, 1)
        elif X.ndim == 1:
            p = newp / newp.sum()
        if log:
            return np.log(p)
        else:
            return p

    @staticmethod
    @np.vectorize
    def _idx2char(index: int) -> str:
        """
        Gives the SMILES character corresponding to a given index.
        The first element is used as a blank character, which is
        not a "real" SMILES character, but is kept for when the
        eigenvalues decay to zero.
        
        Parameters
        ----------
        index : int
            Index to convert into a character
        
        Returns
        -------
        str
            SMILES character
        """
        smi_encoding = [
            " ",
            "H",
            "C",
            "O",
            "N",
            "c",
            "o",
            "n",
            "(",
            ")",
            "[",
            "]",
            ".",
            ":",
            "=",
            "#",
            "\\",
            "/",
            "@",
            "+",
            "-",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        ]
        return smi_encoding[index]

    @staticmethod
    @numba.njit()
    def _beam_search_iteration(
        X: np.ndarray, logp: float, index: int
    ) -> Tuple[float, np.ndarray]:
        """
        Method for iterating through one of the beams. Takes the full array of
        log likelihoods, and the initial seed log likelihood, and iterates
        row by row in the array, finding the index that yields the highest
        cumulative likelihood.
        
        Parameters
        ----------
        x : np.ndarray
            Numpy 2D array containing log likelihoods. Make sure the first
            row is omitted, as it corresponds to an already "chosen" row.
        logp : float
            Log likelihood of the initial seed character
        
        Returns
        -------
        cum_likelihood
            The cumulative loglikelihood for the maximum likelihood chain
        index_chain
            NumPy 1D array containing indices of each character
        """
        nrows, nchar = X.shape
        # cumulative likelihood
        cum_likelihood = logp
        # array for holding the resulting indices
        index_chain = np.zeros(nrows, dtype=np.uint8)
        index_chain[0] = index
        # uniform noise for comparison
        uniform = np.ones(nchar)
        uniform /= nchar
        log_uniform = np.log(uniform)
        # skip the first row
        for row in range(1, nrows):
            # masking prevents numerical overflow; take the larger value of
            # the "real" partition function or a preset minimum value
            # partition_func = max(np.exp(likelihood[likelihood >= -20.]).sum(), 1e-32)
            # Renormalize the likelihoods
            # likelihood = np.log(np.exp(likelihood) / partition_func)
            # Return the index that provides the largest cumulative
            # log likelihood
            likelihood = X[row]
            # If entropy of the chain is sufficiently close to being complete garbage,
            # it's become incoherent and approximates uniform noise. This check basically
            # terminates the chain when the divergence between the likelihood and uniform
            # noise is sufficiently small
            kl_div = np.sum(np.exp(likelihood) * (likelihood - log_uniform))
            if kl_div <= 0.01:
                break
            else:
                max_likelihood = likelihood.max()
                max_index = likelihood.argmax()
                # likelihood = cum_likelihood + X[row]
                cum_likelihood += max_likelihood
                # save the index to the chain
                index_chain[row] = max_index
        # Normalize by chain length
        cum_likelihood /= index_chain.size
        return cum_likelihood, index_chain

    @staticmethod
    def beam_search_decoder(
        X: np.ndarray, width=5, temperature=1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        nsequences, seq_length = X.shape
        # This bit of indexing is somewhat complex; it takes the first line,
        # gets the `width` most probable characters from argsort. Because
        # NumPy sorts in ascending order, the -1 reverses the ordering to
        # make the first `width` elements the largest.
        seed_indices = X[0].argsort()[:width:-1][:width]
        # Adjust likelihoods by temperature
        if temperature != 1.0:
            X = ChainModel._thermalize_likelihoods(X, temperature)
        # upcast for numerical stability
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        X = np.log(X)
        chains = np.zeros((width, nsequences), dtype=np.uint8)
        cum_likelihoods = np.zeros(width, dtype=np.float32)
        # Iterate through each initial seed
        for beam_index, index in enumerate(seed_indices):
            # Get the initial probability and turn into log likelhood
            # the cumulative likelihood becomes a su  mmation
            prob = X[0, index]
            cum_likelihood, index_chain = ChainModel._beam_search_iteration(
                X, prob, index
            )
            cum_likelihoods[beam_index] = cum_likelihood
            chains[beam_index] = index_chain
        # Sort the chains by cumulative log likelihood
        sorting = cum_likelihoods.argsort()[::-1]
        chains = chains[sorting]
        cum_likelihoods = cum_likelihoods[sorting]
        # Translate each row of predictions into SMILES characters
        smiles = np.apply_along_axis(ChainModel._idx2char, -1, chains)
        # Store the results as a dictionary, where each key is the decoded SMILES
        # and the value is the associated likelihood
        decoded_smiles = dict()
        for likelihood, row in zip(cum_likelihoods, smiles):
            characters = "".join(list(row))
            decoded_smiles[smi_sanitizer(characters)] = likelihood
        return decoded_smiles

    @classmethod
    def from_paths(
        cls,
        spec_path: str,
        formula_path: str,
        smiles_path: str,
        functional_path: str,
        cuda=None,
    ):
        """
        Method to create a ChainModel from a set of paths pointing to the
        saved weights of each respective decoder model.

        Returns
        -------
        [type]
            [description]
        """
        if cuda and torch.cuda.is_available():
            kwargs = {"map_location": "cuda"}
        else:
            kwargs = {"map_location": "cpu"}
        spec_encoder = EightPickEncoder()
        spec_encoder.load_state_dict(torch.load(spec_path, **kwargs))
        formula_decoder = EigenFormulaDecoder()
        formula_decoder.load_state_dict(torch.load(formula_path, **kwargs))
        smiles_decoder = EigenSMILESLSTMDecoder()
        smiles_decoder.load_state_dict(torch.load(smiles_path, **kwargs))
        # run inference mode, which sets batch_norm to eval() to get
        # the correct behaviour of batch_norm
        functional_decoder = FunctionalGroupConv(batch_norm=True, inference=True)
        functional_decoder.load_state_dict(torch.load(functional_path, **kwargs))
        chain_model = cls(
            spec_encoder, formula_decoder, smiles_decoder, functional_decoder, cuda=cuda
        )
        if cuda:
            chain_model.cuda()
        return chain_model
