from src.models.torch_models import ChainModel
from typing import Tuple, List
from collections import Counter
from pathlib import Path
import re
import os
import yaml

import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
import periodictable as pt
import joblib
import torch
import numba
from sklearn.neighbors import KernelDensity
from torch.nn import functional as F


def dropout_predict_model(model, inputs: np.ndarray, npredict=10000, dropout=True):
    """
    Function to run a model with dropout predictions. This method sets 
    
    Parameters
    ----------
    model : [type]
        [description]
    inputs : np.ndarray
        Numpy 1D array corresponding to th        e target inputs. This array will
        be tiled across the number of predictions to make a 2D array.
    npredict : int, optional
        [description], by default 500
    dropout : bool, optional
        [description], by default True
    
    Returns
    -------
    [type]
        [description]
    """
    # This sets whether or not dropouts will act even during the testing phase
    K.set_learning_phase(dropout)
    # Make sure all dropout layers think it's dropout time
    for layer in model.layers:
        if "dropout" in layer.name:
            setattr(layer, "training", dropout)
    # This generates copies of the input multiple times
    # such that for any n-dimensional input, we tile it
    # in a new dimension npredict number of times
    shape = (npredict,)
    shape += tuple(1 for i in range(inputs.ndim))
    stretched_inputs = np.tile(inputs, shape)
    # Run the neural network predictions. If dropout is active,
    # the answer should be different each time
    predictions = model.predict(stretched_inputs)
    # predictions = predictions[np.all(predictions >= 0., axis=1)]
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    # Package up the results
    summary = {"mean": mean, "std": std, "full_stack": predictions, "input": inputs}
    return summary


def split_dropout_predict_model(
    model, inputs: np.ndarray, npredict=10000, dropout=True
):
    """
    Function to run a model with dropout predictions. This method sets 
    
    Parameters
    ----------
    model : [type]
        [description]
    inputs : np.ndarray
        Numpy 1D array corresponding to the target inputs. This array will
        be tiled across the number of predictions to make a 2D array.
    npredict : int, optional
        [description], by default 500
    dropout : bool, optional
        [description], by default True
    
    Returns
    -------
    [type]
        [description]
    """
    # This sets whether or not dropouts will act even during the testing phase
    K.set_learning_phase(dropout)
    # Make sure all dropout layers think it's dropout time
    for layer in model.layers:
        if "dropout" in layer.name:
            setattr(layer, "training", dropout)
    stretched_inputs = np.tile(inputs, (npredict, 1))
    # Run the neural network predictions. If dropout is active,
    # the answer should be different each time
    predictions = model.predict(stretched_inputs)
    # This reshapes it into a 3D array; first axis is iteration,
    # second axis corresponds to each model
    predictions = predictions.reshape(npredict, -1, 4)
    # This calculates the average across all iterations
    model_means = predictions.mean(axis=0)
    model_std = predictions.std(axis=0)
    # This calculates the ensemble average, i.e. averaged across
    # all the models and all the iterations
    mean = np.mean(model_means, axis=0)
    std = np.std(model_std, axis=0)
    # Package up the results
    summary = {
        "mean": mean,
        "std": std,
        "full_stack": predictions,
        "input": inputs,
        "model_means": model_means,
        "model_std": model_std,
    }
    return summary


def translate_formula_encoding(prediction: np.ndarray) -> float:
    """
    Calculate the molecular mass from a given chemical formula.
    
    Parameters
    ----------
    prediction : np.ndarray
        [description]
    
    Returns
    -------
    float
        [description]
    """
    formula = ""
    for atom, count in zip(["H", "C", "O", "N"], prediction):
        if count != 0:
            formula += f"{atom}{count}"
    return pt.formula(formula).mass


def sample_smiles_predictions(array: np.ndarray, temperature=0.8) -> str:
    """
    Takes the output of the SMILES LSTM decoder and 
    
    Parameters
    ----------
    array : np.ndarray
        Numpy 1D array of character likelihoods
    temperature : float, optional
        Changes the microstate populations; higher T means more probability
        to less likely characters, by default 0.8
    
    Returns
    -------
    str
        [description]
    """
    smi_list = [
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
    indexes = np.arange(30)
    # Scale with log probability with temperature
    logp = np.log(array) / temperature
    p = np.exp(logp)
    # renormalize to probability mass for each row
    p = p / p.sum(axis=-1)
    return smi_list[np.random.choice(indexes, p=p)]


@numba.njit
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@numba.njit
def batch_sample_smiles_coding(array, temperature=1.0):
    """
    Numba'd version of the SMILES sampling code. Takes a 3D NumPy array
    of float32 dtype, and implements a jit'd loop over the iterations
    and rows, and sampling an index for every row with probability from
    the LSTM output.
    
    Downcasts some of the array types to minimize memory usage.
    
    Parameters
    ----------
    array : [type]
        NumPy 3D array, with shape corresponding to (batchsize, sequences, corpus)
        with a datatype of np.float32.
    temperature : [type], optional
        Same as Boltzmann temperature, by default 1.

    Returns
    -------
    NumPy 2D array
        2D array with sahpe corresponding to (batchsize, sequences)
    """
    iterations, rows, corpus = array.shape
    character_matrix = np.zeros((iterations, rows), dtype=np.uint8)
    for i in range(iterations):
        character_matrix[i] = sample_smiles_coding(array[i], temperature)
    return character_matrix


@numba.njit
def sample_smiles_coding(array, temperature=1.0):
    """
    Numba'd version of the SMILES sampling code. Takes a 3D NumPy array
    of float32 dtype, and implements a jit'd loop over the iterations
    and rows, and sampling an index for every row with probability from
    the LSTM output.
    
    Downcasts some of the array types to minimize memory usage.
    
    Parameters
    ----------
    array : [type]
        NumPy 3D array, with shape corresponding to (batchsize, sequences, corpus)
        with a datatype of np.float32.
    temperature : [type], optional
        Same as Boltzmann temperature, by default 1.
    
    Returns
    -------
    NumPy 2D array
        2D array with sahpe corresponding to (batchsize, sequences)
    """
    indexes = np.arange(0, 30, 1, np.uint8)
    rows, corpus = array.shape
    character_matrix = np.zeros((rows), dtype=np.uint8)
    for i in range(rows):
        p = array[i, :]
        logp = np.log(p) / temperature
        renorm_p = np.exp(logp)
        # renormalize probabilities
        renorm_p = renorm_p / renorm_p.sum()
        character_matrix[i] = rand_choice_nb(indexes, renorm_p)
    return character_matrix


def beam_search_decoder(array, n_best):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in array:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:n_best]
    return sequences


def sanitize_decorator(smi_func) -> str:
    """
    Decorator that will sanitize a predicted SMILES string.
    Basically deletes characters that break SMILES form;
    while it does make most strings work, the current logic
    is not 100% foolproof and some bad codes slip past.
    
    In the future it may be worth running the output of the
    sanitized SMILES through rdkit's parser, and keep generating
    until valid SMILES are produced.
    
    Parameters
    ----------
    smi_func : func
        SMILES predictor function
    
    Returns
    -------
    str
        Sanitized SMILES output
    """

    def sanitized_func(*args, **kwargs):
        smi = smi_func(*args, **kwargs)
        smi = smi_sanitizer(smi)
        return smi

    return sanitized_func


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


def multi_smiles_sampler(
    nsamples: int, array: np.ndarray, temperature=0.8, sanitize=True
) -> List[str]:
    """
    Run a sequence of samples 
    
    Parameters
    ----------
    nsamples : int
        [description]
    array : np.ndarray
        [description]
    temperature : float, optional
        [description], by default 0.8
    
    Returns
    -------
    List[str]
        [description]
    """
    predictor = sample_smiles_predictions
    if sanitize:
        predictor = sanitize_decorator(predictor)
    smi_list = [predictor(array, temperature) for i in range(nsamples)]
    unique = list(set(smi_list))
    print(f"Generated {len(unique)} SMILES codes.")
    return unique


def idx2char(index: int) -> str:
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


def convert_predictions(
    predictions: np.ndarray, bins=40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a mass spectrum for the predicted chemical formula. Takes the
    full set of predictions generated by the `dropout_predict_model`
    
    Parameters
    ----------
    predictions : np.ndarray
        A NumPy 2D array, where each row corresponds to a molecule, and the
        columns correspond to the number of atoms for H, C, O, N
    bins : int or np.ndarray, optional
        Bin specification for the histograms, by default 40
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Histogram and bin edges for the mass spectrum
    """
    assert predictions.ndim == 2
    masses = calculate_mass_distribution(predictions)
    histo, bins = np.histogram(masses, bins=bins)
    # Get the probability distribution function
    histo = histo / histo.sum()
    return histo, bins[:-1]


def calculate_kde_probability(
    Y: np.ndarray, X=None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the kernel density estimate of a set of samples/predictions.
    This function basically replaces the use of histograms for estimating
    quantities, and overall makes figures look a bit nicer.
    
    Parameters
    ----------
    Y : np.ndarray
        [description]
    bandwidth : float, optional
        [description], by default 1.0
    X : [type], optional
        [description], by default None
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        [description]
    """
    if X is None:
        # If no X axis given to evaluate probability, then
        # generate one. Extra empty axis is used because
        # sklearn estimator expects 2D array
        X = np.linspace(Y.min(), Y.max(), 200)
    if X.ndim != 2:
        X = X[:, None]
    # kwargs are passed into the creation of KernelDensity
    # fitting object
    default_settings = {"bandwidth": 1.0, "kernel": "gaussian"}
    default_settings.update(**kwargs)
    kde_obj = KernelDensity(**default_settings)
    # sklearn estimator expects 2D array
    if Y.ndim != 2:
        Y = Y[:, None]
    _ = kde_obj.fit(Y)
    p = np.exp(kde_obj.score_samples(X))
    # make sure probability is normalized
    p = p / p.sum()
    return (X, p)


def calculate_mass_distribution(predictions: np.ndarray, quantize=True) -> np.ndarray:
    """
    Calculate the mass of every composition predicted in a batch of predictions.
    This method works as a NumPy multiplication of number of atoms and its
    corresponding mass - the `translate_formula_encoding` function does
    something similar, but is not intended for operations on large arrays.
    
    Parameters
    ----------
    predictions : np.ndarray
        A NumPy 2D array, where each row corresponds to a molecule, and the
        columns correspond to the number of atoms for H, C, O, N
    
    Returns
    -------
    np.ndarray
        NumPy 1D array containing the summed molecule mass
    """
    if quantize:
        predictions = np.round(predictions, 0)
    masses = np.apply_along_axis(translate_formula_encoding, -1, predictions)
    return masses


def simulate_mass_spectrum(
    predictions: np.ndarray, quantize=True, X=None, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic mass spectrum based on the predicted number of atoms.
    The input is a 2D array which is fed into `calculate_mass_distribution`.
    
    Parameters
    ----------
    predictions : np.ndarray
        [description]
    quantize : bool, optional
        [description], by default True
    
    Returns
    -------
    mass_bins, mass_p
        Kernel density estimate of the predicted masses.
    """
    masses = calculate_mass_distribution(predictions, quantize)
    mass_bins, mass_p = calculate_kde_probability(masses, X=X, **kwargs)
    return (mass_bins, mass_p)


def summary_statistics(
    X: np.ndarray, Y: np.ndarray, q=[0.05, 0.5, 0.95]
) -> Tuple[float, float, np.ndarray]:
    """
    For a given set of bins (X) and probabilities (Y), calculate some summary
    statistics including quantile/density ranges, the expectation value, and
    the maximum likelihood estimate.
    
    Parameters
    ----------
    X : np.ndarray
        Numpy 1D array containing the quantity to be estimated
    Y : np.ndarray
        Numpy 1D array corresponding to the probability of X
    q : list, optional
        List of floats between 0 and 1, corresponding to the quantile ranges
        to evaluate the posterior density estimates with; by default [0.05, 0.5, 0.95],
        which correspond to the 95% credible interval and the median
    
    Returns
    -------
    Tuple[float, float, np.ndarray]
        [description]
    """
    # Make sure probability is density
    Y = Y / Y.sum()
    # compute the cumulative density function
    cdf = np.cumsum(Y)
    cdf = cdf / cdf.max()
    # Get the indices of where the cumulative density matches
    indices = np.searchsorted(cdf, q, side="right")
    quantile_values = X[indices]
    # calculate the expectation value
    if X.ndim == 2:
        X = X.flatten()
    expec = np.sum(Y * X)
    # calculate the maximum likelihood estimate
    mle = X[np.argmax(Y)]
    return (mle, expec, quantile_values)


def calculate_prediction_statistics(predictions: np.ndarray, q=[0.05, 0.95]):
    stats = list()
    x = np.linspace(0.0, 20.0, 200)
    for index, atom in enumerate(["H", "C", "O", "N"]):
        # Calculate the quantile ranges for each atom
        q = np.sort(q)
        quantile_range = np.quantile(predictions[:, index], q=q)
        min_q = np.min(quantile_range)
        max_q = np.max(quantile_range)
        # Compute the kernel density estimate to determine expectation
        # value and other stuff
        kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        kde.fit(predictions[:, index][:, None])
        pde = np.exp(kde.score_samples(x[:, None]))
        pde /= pde.sum()
        # Calculate the probability density function
        expec = np.sum(pde * x)
        # Calculate the variance
        variance = np.sum(x ** 2.0 * pde) - expec ** 2.0
        std_dev = np.sqrt(variance)
        stats.append(
            {
                "expec": expec,
                "variance": variance,
                "std": std_dev,
                f"Q{int(q[0] * 100.)}%": min_q,
                f"Q{int(q[1] * 100.)}%": max_q,
            }
        )
    df = pd.DataFrame(stats, index=["H", "C", "O", "N"])
    return df


def timeshift_array(array: np.ndarray, window_length=4, n_timeshifts=100) -> np.ndarray:
    """
    Uses a neat NumPy trick to vectorize a sliding operation on a 1D array.
    Basically uses a 2D indexer to generate n_shifts number of windows of
    n_elements length, such that the resulting array is a 2D array where
    each successive row is shifted over by one.
    
    The default values are optimized for a maximum of 30 atoms.
    
    This is based off this SO answer:
    https://stackoverflow.com/a/42258242
    
    Parameters
    ----------
    array : np.ndarray
        [description]
    n_elements : int
        Length of each window, by default 5
    n_timeshifts : int, optional
        Number of timeslices, by default 100
    
    Returns
    -------
    np.ndarray
        NumPy 2D array with rows corresponding to chunks of a sliding
        window through the input array
    """
    shifted = np.zeros((n_timeshifts, window_length), dtype=np.float32)
    n_actual = array.size - window_length + 1
    indexer = np.arange(window_length).reshape(1, -1) + np.arange(n_actual).reshape(
        -1, 1
    )
    shifted[:n_actual, :] = array[indexer]
    return shifted


class MoleculeIdentifier:
    """
    Function written in Python that wraps the Tensorflow models that
    have been pre-trained.
    
    Raises
    ------
    ValueError
        [description]
    """

    def __init__(
        self,
        encoders: List[str],
        formula_decoders: List[str],
        smiles_decoders: List[str],
        dropout=True,
        loader=keras.models.load_model,
        loader_args={},
    ):
        self._encoders = [
            loader(encoder_path, **loader_args) for encoder_path in encoders
        ]
        self._formula_decoders = [
            loader(decoder_path, **loader_args) for decoder_path in formula_decoders
        ]
        self._smiles_decoders = [
            loader(decoder_path, **loader_args) for decoder_path in smiles_decoders
        ]
        # Flag for determining if sampling is deterministic
        self.dropout = dropout
        # Attributes for holding the probability distributions and the generated
        # SMILES strings respectively
        self.smiles_coding = list()
        self.smiles = list()

    @classmethod
    def from_yml(cls, yml_path: str):
        """
        Method of creating a `MoleculeIdentifier` object via settings written
        down in a YAML file.
        
        The minimal amount of information required in the YAML file are paths
        to each of the tensorflow models.
        
        Returns
        -------
        MoleculeIdentifier
            Instance of a `MoleculeIdentifier` object
        """
        settings = {"dropout": True}
        with open(yml_path, "r") as read_file:
            temp = yaml.safe_load(read_file)
            settings.update(**temp)
        detective = cls(**settings)
        return detective

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    def predict(self, X: np.ndarray, niter=1000, **kwargs):
        """
        All of the predictions are vectorized, meaning they should be
        on the order of milliseconds quick to do 1000 or so predictions.
        
        Parameters
        ----------
        X : np.ndarray
            [description]
        niter : int, optional
            [description], by default 1000
        
        Raises
        ------
        ValueError
            [description]
        """
        # First check to make sure the input data is correct
        if X.ndim != 1 or X.size != 8:
            raise ValueError(
                "Dimensionality or length of X is incorrect; "
                "please provide a 1D array that is 8 elements long."
            )
        X_input = np.tile(X, (niter, 1))
        # Encoding constants into eigenspectra
        spectra = np.zeros(shape=(len(self._encoders), niter, 30), dtype=np.float32)
        for index, model in enumerate(self._encoders):
            spectra[index, :, :] = model.predict(X_input)
        # Formula decoding
        formula = np.zeros(
            shape=(len(self._formula_decoders), niter, 4), dtype=np.float32
        )
        for index, model in enumerate(self._formula_decoders):
            formula[index, :, :] = model.predict(spectra[index, :, :])
        self.formulas = formula
        # Flatten the separate encoder models, and just have a stack of spectra
        flat_spectra = np.reshape(spectra, (len(self._encoders) * niter, 30))
        # Create the sliding windows for every spectrum
        timeshifted_spectra = np.apply_along_axis(timeshift_array, -1, flat_spectra)
        smiles_coding = self._smiles_decoder.predict(timeshifted_spectra)
        self.smiles_coding = smiles_coding
        self.smiles = self.translate_smiles(smiles_coding, **kwargs)
        return self.smiles

    def translate_smiles(
        self, smiles_coding=None, temperature=1.0, sanitize=True, unique=True,
    ) -> List[str]:
        """
        Function that will sample the probability distributions generated
        by the LSTM model, and generate corresponding SMILES codes.
        
        The `temperature` parameter adjusts how radical the codes can be,
        with larger temperatures yielding increasingly crazy SMILES codes.
        The `sanitize` flag determines if the outputs of these strings
        are cleaned up after the fact, removing characters or patterns
        that I've encountered before that make otherwise working strings
        invalid.
        
        This function is called at the end of `predict`; once that's been
        done once, you can repeatedly/iteratively call this function
        to generate additional SMILES strings, as this step is probabilistic.
        
        Parameters
        ----------
        smiles_coding : int, np.ndarray, or None optional
            If an integer is supplied, this value is used to index which model
            to sample from. If a NumPy array is provided, the supplied values
            will be used to sample from. If None (default), the first group
            will automatically be used.
        temperature : float, optional
            Parameter that adjusts the craziness of generated strings. 
            Lower values mean less crazy; by default 1.0
        sanitize : bool, optional
            If True, uses a decorated function that will clean up
            invalid strings.
        
        Returns
        -------
        List[str]
            List of generated SMILES strings
        """
        # If the coding supplied is in integer, use it to index
        # which probability distribution to sample from
        if type(smiles_coding) == int:
            smiles_coding = self.smiles_coding[smiles_coding]
        elif type(smiles_coding) == np.ndarray:
            smiles_coding = smiles_coding
        else:
            # default option is to take group 0
            smiles_coding = self.smiles_coding[0]
        if smiles_coding.ndim == 2:
            predictor = sample_smiles_coding
        elif smiles_coding.ndim == 3:
            predictor = batch_sample_smiles_coding
        vect_idx2char = np.vectorize(idx2char)
        indices = predictor(smiles_coding, temperature=temperature)
        smiles_characters = vect_idx2char(indices)
        smiles = [self._smi_helper(sequence) for sequence in smiles_characters]
        if unique:
            smiles = list(set(smiles))
        else:
            smiles = Counter(smiles)
        return smiles

    def _smi_helper(self, array: np.ndarray, sanitize=True):
        smi_string = "".join(list(array))
        if sanitize:
            try:
                if len(smi_string) > 1:
                    smi_string = smi_sanitizer(smi_string)
                return smi_string
            except IndexError:
                print(f"{smi_string} failed!")
        return smi_string


class KerasIdentifier(MoleculeIdentifier):
    def __init__(
        self,
        encoders,
        formula_decoders,
        smiles_decoders,
        dropout=True,
        nthreads=4,
        debug=False,
        loader=keras.models.load_model,
        loader_args={"custom_objects": {"LeakyReLU": keras.layers.LeakyReLU}},
    ):
        # Performance considerations, which setup parallelism for the
        # tensorflow operations
        os.environ["OMP_NUM_THREADS"] = str(nthreads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(nthreads)
        if debug:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        super().__init__(
            encoders,
            formula_decoders,
            smiles_decoders,
            dropout=dropout,
            loader=loader,
            loader_args=loader_args,
        )

    def predict(self, X: np.ndarray, niter=1000, dropout=None, **kwargs):
        """
        All of the predictions are vectorized, meaning they should be
        on the order of milliseconds quick to do 1000 or so predictions.
        
        Parameters
        ----------
        X : np.ndarray
            [description]
        niter : int, optional
            [description], by default 1000
        
        Raises
        ------
        ValueError
            [description]
        """
        if not dropout:
            dropout = self.dropout
        K.set_learning_phase(dropout)
        # First check to make sure the input data is correct
        if X.ndim != 1 or X.size != 8:
            raise ValueError(
                "Dimensionality or length of X is incorrect; "
                "please provide a 1D array that is 8 elements long."
            )
        X_input = np.tile(X, (niter, 1))
        # Encoding constants into eigenspectra
        spectra = np.zeros(shape=(len(self._encoders), niter, 30), dtype=np.float32)
        for index, model in enumerate(self._encoders):
            spectra[index, :, :] = model.predict(X_input)
        # Formula decoding
        formula = np.zeros(
            shape=(len(self._formula_decoders), niter, 4), dtype=np.float32
        )
        for index, model in enumerate(self._formula_decoders):
            formula[index, :, :] = model.predict(spectra[index, :, :])
        self.formulas = formula
        # Flatten the separate encoder models, and just have a stack of spectra
        flat_spectra = np.reshape(spectra, (len(self._encoders) * niter, 30))
        # Create the sliding windows for every spectrum
        timeshifted_spectra = np.apply_along_axis(timeshift_array, -1, flat_spectra)
        smiles_coding = self._smiles_decoder.predict(timeshifted_spectra)
        self.smiles_coding = smiles_coding
        self.smiles = self.translate_smiles(smiles_coding, **kwargs)
        return self.smiles


class TorchIdentifier(MoleculeIdentifier):
    def __init__(
        self,
        encoders,
        formula_decoders,
        smiles_decoders,
        dropout=True,
        loader=torch.load,
        loader_args={},
    ):
        super().__init__(
            encoders,
            formula_decoders,
            smiles_decoders,
            dropout=dropout,
            loader=loader,
            loader_args=loader_args,
        )
        # Disable gradient computation which is not needed
        torch.set_grad_enabled(False)

    def __call__(self, inputs: np.ndarray, model_index=0):
        return self.forward(inputs, model_index)

    def predict(self, X: np.ndarray, niter=1000, dropout=None, **kwargs):
        """
        All of the predictions are vectorized, meaning they should be
        on the order of milliseconds quick to do 1000 or so predictions.
        
        This has minor differences from the tensorflow version, as the
        prediction syntax is different; the `torch` version has a functional
        syntax compared to the `tensorflow` `model.predict()` object method.
        
        Parameters
        ----------
        X : np.ndarray
            [description]
        niter : int, optional
            [description], by default 1000
        
        Raises
        ------
        ValueError
            [description]
        """
        # Context manager for no gradient computation
        with torch.no_grad():
            if dropout is None:
                dropout = self.dropout
            # First check to make sure the input data is correct
            if X.ndim != 1 or X.size != 8:
                raise ValueError(
                    "Dimensionality or length of X is incorrect; "
                    "please provide a 1D array that is 8 elements long."
                )
            X_input = np.tile(X, (niter, 1))
            # Encoding constants into eigenspectra
            spectra = np.zeros(shape=(len(self._encoders), niter, 30), dtype=np.float32)
            for index, model in enumerate(self._encoders):
                # Set inference mode
                model.train(dropout)
                # Run input through models, and convert to numpy arrays
                spectra[index, :, :] = model(X_input).numpy()
            self.spectra = spectra
            # Formula decoding
            formula = np.zeros(
                shape=(len(self._formula_decoders), niter, 4), dtype=np.float32
            )
            for index, model in enumerate(self._formula_decoders):
                # Set inference mode
                model.train(dropout)
                formula[index, :, :] = model(spectra[index, :, :]).numpy()
            self.formulas = formula
            # Get summary statistics for the formula predictions
            self.summarize_formula()
            # Attribute for storing SMILES strings based on predictions
            self.smiles = list()
            self.smiles_coding = np.zeros((4, niter, 100, 30), dtype=np.float32)
            # Run through the SMILES decoding
            for index, model in enumerate(self._smiles_decoders):
                # Set inference mode
                model.train(dropout)
                # Create the sliding windows for every spectrum
                timeshifted_spectra = np.apply_along_axis(
                    timeshift_array, -1, spectra[index, :, :]
                )
                # Apply softmax to the predictions; shape is (niter, 100, 30)
                # softmax normalizes likelihoods across rows
                smiles_coding = F.softmax(model(timeshifted_spectra), dim=-1).numpy()
                # Downcast to compress the result
                self.smiles_coding[index] = smiles_coding.astype(np.float16)
                smiles = self.translate_smiles(index)
                # Take the the mean estimate and use that to sample SMILES
                self.smiles.append(smiles)
            return self.smiles

    def _smi_helper(self, array: np.ndarray):
        smi_string = "".join(list(array))
        try:
            if len(smi_string) > 1:
                smi_string = smi_sanitizer(smi_string)
            return smi_string
        except IndexError:
            print(smi_string)
            return ""

    def summarize_formula(self, q=[0.05, 0.5, 0.95]):
        # Formalize some statistics about the sampling
        mass_stats = list()
        mass_spectra = list()
        formula_dict = dict()
        atoms = ["H", "C", "O", "N"]
        for index, model in enumerate(self._formula_decoders):
            # Calculate the mass properties
            mass_bins, mass_p = simulate_mass_spectrum(
                self.formulas[index], X=np.linspace(0.0, 200.0, 300, dtype=np.float16)
            )
            mass_spectra.append(mass_p)
            mle, expec, q_values = summary_statistics(mass_bins, mass_p, q=q)
            mass_stats.append({"MLE": mle, "expec": expec, "Q": q_values})
            for atom_index, symbol in enumerate(atoms):
                formula_bins, formula_p = calculate_kde_probability(
                    self.formulas[index, :, atom_index],
                    X=np.linspace(0.0, 20.0, 200, dtype=np.float16),
                )
                mle, expec, q_values = summary_statistics(formula_bins, formula_p, q=q)
                formula_dict[(f"model{index}", symbol)] = {
                    "MLE": mle,
                    "Expec": expec,
                    "Quantiles": q_values,
                }
        self.mass_summary = mass_stats
        self.mass_spectra = mass_spectra
        self.mass_bins = mass_bins
        self.formula_bins = formula_bins
        self.formula_df = pd.DataFrame(formula_dict)

    def forward(
        self, inputs: np.ndarray, model_index=0, temperature=1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a simple forward pass of one chain of models. Takes a
        NumPy 1D array of spectroscopic parameters, and returns the
        eigenspectrum, formula, and SMILES encodings.
        
        Returns
        -------
        [type]
            [description]
        """
        assert model_index <= len(self._encoders)
        with torch.no_grad():
            # Retrieve the corresponding model from each index
            encoder = self._encoders[model_index]
            formula_decoder = self._formula_decoders[model_index]
            smi_decoder = self._smiles_decoders[model_index]
            spectrum = encoder(inputs)
            formula = formula_decoder(spectrum)
            # Create the timeshift windows
            shifted_spectra = timeshift_array(spectrum.numpy())
            # Apply softmax to the output
            smiles_encoding = F.softmax(smi_decoder(shifted_spectra[:, None]), dim=-1)
            # Get rid of intermediate dimension
            smiles_encoding.squeeze_(1)
            smiles = sample_smiles_predictions(smiles_encoding, temperature)
            smiles = smi_sanitizer(smiles)
            # smiles = self.translate_smiles(smiles_encoding.numpy(), **kwargs)
            return spectrum, formula, smiles_encoding, smiles


class InferenceMachine:
    def __init__(self, model_root):
        if type(model_root) == str:
            model_root = Path(model_root)
        self.models = list()
        for i in range(4):
            self.models.append(
                ChainModel.from_paths(
                    model_root.joinpath(f"/Split-SpecEncoder-group{i}.pt"),
                    model_root.joinpath(f"/Split-FormulaDecoder-group{i}.pt"),
                    model_root.joinpath(f"/Split-EigenSMILESLSTMDecoder-group{i}.pt"),
                    model_root.joinpath(f"/Split-FunctionalDecoder-group{i}.pt")
                )
            )
        self.spectra = None
        self.formula = None
        self.smiles = None
        self.smi_histogram = None
        self.functional_groups = None

    def __call__(self, inputs, npredict=1000, temperature=1.0):
        return self.run_prediction(inputs, npredict, temperature)

    def run_prediction(self, inputs, npredict=1000, temperature=1.0):
        spectra_array = np.zeros((4, npredict, 30), dtype=np.float32)
        formula_array = np.zeros((4, npredict, 4), dtype=np.float16)
        functional_array = np.zeros((4, 23, 100), dtype=np.float16)
        smiles = dict()
        smi_histogram = Counter()
        for index, model in enumerate(self.models):
            spectra, formula, encoded_smi, functional = model(inputs, npredict)
            spectra_array[index] = spectra
            formula_array[index] = formula
            # Convert the functional group data into a heatmap of likelihoods
            # with kernel density estimation
            heatmap = model.functional_group_kde_analysis(functional)
            functional_array[index] = heatmap
            # Use beam decoder on smi encodings
            for smi in encoded_smi:
                decoded = model.beam_search_decoder(smi, temperature=temperature)
                smiles.update(**decoded)
                unique_smi = Counter(list(decoded.keys()))
                smi_histogram += unique_smi
        self.spectra = spectra_array
        self.formula = formula_array
        self.smiles = smiles
        self.smi_histogram = smi_histogram
        self.functional_groups = functional_array
        return spectra_array, formula_array, smiles, smi_histogram, functional_array

    def unknown_sampling(
        self,
        inputs,
        micro_iterations=500,
        macro_iterations=100,
        noise_multiplier=[1, 1, 1, 1, 1, 1, 1, 1],
    ):
        comp_labels = ["HC", "HCO", "HCN", "HCNO"]
        for macro_idx in range(macro_iterations):
            noise = np.random.normal(size=(8)) * np.array(noise_multiplier)
            perturbed_inputs = inputs + noise
            spectra, formula, smiles, smi_histogram = self.run_prediction(
                perturbed_inputs, micro_iterations
            )
            summary = dict()
            for comp_idx, comp_labels in enumerate(comp_labels):
                df = calculate_prediction_statistics(formula[comp_idx])
