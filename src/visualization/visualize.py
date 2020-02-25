"""
visualize.py

Include routines here that will visualize parts of your analysis.

This can include taking serialized models and seeing how a trained
model "sees" inputs at each layer, as well as just making figures
for talks and writeups.
"""

from typing import List, Dict
from itertools import product
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from tensorflow.keras import Model
from palettable.cartocolors import sequential as pts
from palettable.cartocolors import qualitative as ptq

from src.models import predict_model


def create_intermediate_layers(model):
    layered_models = list()
    for layer in model.layers:
        layered_models.append(Model(inputs=model.input, outputs=layer.output))
    return layered_models


def intermediate_layer_prediction(layered_models: List, input: np.ndarray):
    outputs = [layer.predict(input) for layer in layered_models]
    return outputs


def get_weights_biases(model):
    weights = dict()
    biases = dict()
    x = [np.linspace(-1.0, 1.0, model.inputs[0].shape[1])]
    # Get the weights and biases from the model
    for layer in model.trainable_weights:
        if "batch" not in layer.name:
            if "dense" in layer.name and "kernel" in layer.name:
                weights[layer.name] = np.abs(layer.numpy())
                x.append(np.linspace(-1.0, 1.0, layer.numpy().shape[1]))
            if "dense" in layer.name and "bias" in layer.name:
                biases[layer.name] = layer.numpy()
    norm_weights = minmax_scaling(weights)
    norm_biases = minmax_scaling(biases, min_val=0.3, max_val=1.0)
    return weights, biases, norm_weights, norm_biases, x


def visualize_model(model):
    nlayers = len([layer for layer in model.layers if "batch" not in layer.name])
    weights, biases, norm_weights, norm_biases, x = get_weights_biases(model)
    y = np.arange(len(x))[::-1]
    colors = plt.cm.Pastel1(np.linspace(0.0, 1.0, nlayers))
    fig, ax = plt.subplots()
    # Loop over every single x/y position - this part of the code
    # plots lines between each hidden unit, with the alpha corresponding
    # to the associated weight between two units
    for index, (x_val, y_val, weight_set, bias_set) in enumerate(
        zip(x, y, norm_weights.values(), norm_biases.values())
    ):
        # try/except because the last layer doesn't have
        # a layer in front of it
        next_x = x[index + 1]
        next_y = y[index + 1]
        y_pair = (y_val, next_y)
        # Loop over each pair of x; current and next layer, and draw
        # line between the nodes according to their weights/biases
        # This is incredibly poor form, but I want the code working first
        # writing it as a nested for loop over arrays
        for i in range(x_val.size):
            for j in range(next_x.size):
                x_pair = (x_val[i], next_x[j])
                y_pair = (y_val, next_y)
                _ = ax.plot(
                    x_pair,
                    y_pair,
                    color=pts.SunsetDark_7.mpl_colormap(weight_set[i, j]),
                    zorder=1,
                    alpha=bias_set[j],
                    lw=0.5,
                )
    for index, x_val in enumerate(x):
        ax.scatter(
            x_val,
            [y[index]] * x_val.size,
            zorder=5,
            color=colors[index],
            edgecolor="k",
            lw=0.5,
        )

    ax.set_xlim([-1.5, 1.5])
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    return fig, ax, norm_weights, norm_biases


def minmax_scaling(
    arrays_dict: Dict[str, np.ndarray], min_val=0.0, max_val=1.0
) -> Dict[str, np.ndarray]:
    """
    Performs the same kind of minmax scaling as the sklearn
    MinMaxScaler class. This ideally gets all of the features into
    the range of 0 to 1.
    
    Parameters
    ----------
    arrays_dict : Dict[str, np.ndarray]
        Dictionary of arrays, where the key corresponds to
        the name of the layer, and the arrays are 1D arrays
        of weights or biases
    
    Returns
    -------
    Dict[str, np.ndarray]
        [description]
    """
    new_arrays = dict()
    # Loop over the weights/biases of each layer, and scale
    # the values to MinMax
    for key, array in arrays_dict.items():
        scale = (max_val - min_val) / (array.max(axis=0) - array.min(axis=0))
        new_arrays[key] = scale * array + min_val - array.min(axis=0) * scale
    return new_arrays


def one_scaling(arrays_dict):
    new_arrays = copy(arrays_dict)
    X_max = 0.0
    for array in arrays_dict.values():
        if array.max() > X_max:
            X_max = array.max()
    for key, array in arrays_dict.items():
        temp = array / X_max
        # temp[temp < 0.2] = 0.2
        new_arrays[key] = temp
    return new_arrays


def plot_model_performance(test_data_dict, key: str, ax=None):
    """
    Function to make a bar plot comparison of predictions against
    the actual data. The `test_data_dict` corresponds to the
    subdictionary produced from the training phase, which tests
    the model against three known formula, specifically the dictionary
    within the "test" key.
    
    For example, test_data_dict = model_data["test]
    
    Parameters
    ----------
    test_data_dict : dict
        "test" dictionary contained within a summary pickle from
        the training phase.
    key : str
        Name of the molecule to plot up within `test_data_dict`
    ax : matplotlib axis object, or None; optional
        If an axis object is provided, the plots will be made
        using the provided axis. Otherwise, a new matplotlib
        figure/axis object will be generated.
    
    Returns
    -------
    fig, ax
        matplotlib figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots()

    test_data = test_data_dict[key]

    x = np.arange(test_data["mean"].size)

    # Draw scatter points with some random noise added to
    # the x values to visualize the spread in sampled y
    split_arrays = [
        test_data["full_stack"][:, index]
        for index in range(test_data["full_stack"].shape[-1])
    ]
    violin = ax.violinplot(
        split_arrays, showmeans=False, showmedians=False, showextrema=False
    )
    for pc in violin["bodies"]:
        pc.set_facecolor("#4989a0")
        pc.set_edgecolor("k")
        pc.set_linewidth(1.0)
        pc.set_alpha(0.6)

    # Draw out the summary statistics
    hp5, median, hp95 = np.quantile(
        test_data["full_stack"], q=[0.05, 0.5, 0.95], axis=0
    )
    ax.scatter(x + 1, hp5, marker="_", lw=2.0, s=120.0, c="k", zorder=10)
    ax.scatter(
        x + 1, hp95, marker="_", lw=2.0, s=120.0, c="k", zorder=10,
    )
    ax.scatter(
        x + 1, test_data["mean"], c="#254d6c", s=75.0, zorder=10, lw=1.0, edgecolor="k"
    )

    ax.bar(
        x + 1,
        test_data["target"][::-1],
        alpha=0.7,
        label="Target",
        color="#ec6866",
        edgecolor="k",
        lw=0.5,
        width=0.2,
        align="center",
    )

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim([0.0, 12.0])
    # ax.legend()
    # ax.axhline(0.0)
    try:
        return fig, ax
    except UnboundLocalError:
        return ax


def plot_composition(
    predictions, actual=None, bins=np.linspace(0.0, 20.0), q=[0.05, 0.95]
):
    """
    Create a four panel plot for the likelihood distributions for the number
    of atoms of H, C, O, N.
    
    Parameters
    ----------
    predictions : [type]
        [description]
    actual : [type], optional
        [description], by default None
    bins : [type], optional
        [description], by default np.linspace(0., 20.)
    
    Returns
    -------
    [type]
        [description]
    """
    stats_df = predict_model.calculate_prediction_statistics(predictions, q=q)
    fig, axarray = plt.subplots(1, 4, figsize=(8, 2.2), sharey=True)

    labels = ["H", "C", "O", "N"]

    for index, label in enumerate(labels):
        # Each axis corresponds to an atom; here we're gathering up
        # the predictions from each model for a given atom
        ax = axarray[index]
        violin = ax.violinplot(
            predictions, showmeans=False, showmedians=False, showextrema=False
        )
        for pc in violin["bodies"]:
            pc.set_facecolor("#4989a0")
            pc.set_edgecolor("k")
            pc.set_linewidth(1.0)
            pc.set_alpha(0.6)

        # Draw out the summary statistics
        hp5, median, hp95 = np.quantile(predictions, q=[0.05, 0.5, 0.95], axis=0)
        ax.scatter(x + 1, hp5, marker="_", lw=2.0, s=120.0, c="k", zorder=10)
        ax.scatter(
            x + 1, hp95, marker="_", lw=2.0, s=120.0, c="k", zorder=10,
        )
        # ax.scatter(
        #     x + 1, predictions["mean"], c="#254d6c", s=75.0, zorder=10, lw=1.0, edgecolor="k"
        # )
        # _ = ax.bar(actual_bins, histo, alpha=0.5, width=0.4)
        expec, variance, std, min_q, max_q = stats_df.loc[label].to_numpy()
        # Annotate the point statistics
        ax.text(0.5, 0.85, f"${expec:.1f} \pm {std:.1f}$", transform=ax.transAxes)
        if actual is not None:
            ax.axvline(actual[index], lw=1.5, ymax=0.79)
    return fig, axarray


def plot_split_composition(
    predictions,
    actual=None,
    bins=np.linspace(0.0, 20.0),
    q=[0.05, 0.95],
    axarray=None,
    title=None,
):
    """
    Create a four panel plot for the likelihood distributions for the number
    of atoms of H, C, O, N.
    
    Parameters
    ----------
    predictions : [type]
        [description]
    actual : [type], optional
        [description], by default None
    bins : [type], optional
        [description], by default np.linspace(0., 20.)
    
    Returns
    -------
    [type]
        [description]
    """
    n_model, n_predict, n_features = predictions.shape
    if axarray is None:
        fig, axarray = plt.subplots(1, 4, figsize=(10.0, 3.0))
    else:
        # Assume we have enough axes to plot with
        assert len(axarray) == n_features
    labels = ["H", "C", "O", "N"]
    model_colors = ptq.Safe_4.hex_colors
    x = np.arange(len(labels))
    # For each atom, plot out each model output separately
    for model_index in range(n_model):
        ax = axarray[model_index]
        atom_arrays = predictions[model_index]
        violin = ax.violinplot(
            atom_arrays, showmeans=False, showmedians=False, showextrema=False
        )
        # Make each violin color correspond to the model
        for pc in violin["bodies"]:
            pc.set_facecolor(model_colors[model_index])
            pc.set_edgecolor("k")
            pc.set_linewidth(1.0)
            pc.set_alpha(0.6)
            pc.set_zorder(3)
        hp5, median, hp95 = np.quantile(atom_arrays, [0.05, 0.5, 0.95], axis=0)
        ax.scatter(x + 1, hp5, marker="_", s=200.0, c="k")
        ax.scatter(x + 1, hp95, marker="_", s=200.0, c="k")
        ax.scatter(x + 1, median, marker="o", s=50.0, c="k", alpha=0.8)
        ax.set_xticks([])
        for spine in ["top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        if actual is not None:
            ax.axhline(actual[atom_index], lw=1.5, ls="--")
        ax.set_ylabel("Number of atoms")
    if title:
        fig.suptitle(title)
    try:
        return fig, axarray
    except UnboundLocalError:
        return axarray
