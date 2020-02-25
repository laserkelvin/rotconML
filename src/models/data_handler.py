"""
data_handler.py

This module implements the DataGenerator class that handles
all of the passing of data from disk to Keras.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from numba import jit, vectorize, float64


class DataGenerator(Sequence):
    """
    Object for handling training data to feed to a Keras model.
    """

    def __init__(
        self,
        h5_group,
        X_label,
        Y_label,
        batch_size=32,
        shuffle=True,
        augment=False,
        sigma=1.0,
        seed=None,
        indexes=None,
        aug_distribution=None,
    ):
        self.batch_size = batch_size
        self.h5_obj = h5_group
        self.shuffle = shuffle
        self.augment = augment
        self.X_label = X_label
        self.Y_label = Y_label
        self.sigma = sigma
        # Indexes for shuffling
        if indexes is None:
            self.indexes = np.arange(len(self.h5_obj[X_label]))
        else:
            self.indexes = indexes
        self.on_epoch_end()
        # Get the expected shape of the data
        self.X_shape = self.h5_obj[X_label].shape
        self.Y_shape = self.h5_obj[Y_label].shape
        # For 3D arrays (i.e. images) we will reshape the array
        if self.h5_obj[Y_label].ndim == 3:
            self.Y_shape = tuple([Y ** 2 for Y in self.Y_shape])
        if self.h5_obj[X_label].ndim == 3:
            self.X_shape = tuple([Y ** 2 for Y in self.X_shape])
        if aug_distribution is not None:
            augment_sample_df = pd.read_csv(aug_distribution)
            self.__X_shifts = augment_sample_df["Bins"].values
            self.__p_shifts = augment_sample_df["Modeled"].values

    def on_epoch_end(self):
        """
        If required, this step will shuffle the indexes to mix
        up the data set.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __augment_noise__(self, a: np.ndarray):
        """
        Function to augment data with Gaussian noise, with a specified
        amount of "blurring" centered at zero.
        
        Parameters
        ----------
        a : np.ndarray
            Array to be augmented
        sigma : float, optional
            Width of the Gaussian noise, by default 1.
        
        Returns
        -------
        np.ndarray
            Augmented array
        """
        sigma = np.average(a) / 2.0
        noise = np.random.normal(loc=0.0, scale=sigma, size=a.shape)
        aug_a = a + noise
        return aug_a

    def augment_rotational_constants(
        self, X: np.ndarray, shifts: np.ndarray, p: np.ndarray
    ):
        """
        Function to augment the rotational constants based on the
        uncertainty of the rotational constants. The objective of
        this is to try and incorporate the uncertainty of the
        theoretical rotational constants in our dataset, whilst
        also helping by increasing the potential number of training
        and validation data.
        
        The expected shifts from the experimental data are taken
        from the Bayesian sampling paper for a particular method/basis.
        These shifts are randomly sampled with probability corresponding
        to their posterior predictions.
        
        This function is not jit'd with numba because the `p` argument
        in `np.random.choice` is not yet supported.
        
        Parameters
        ----------
        X : np.ndarray
            X data to augment; assumes the rotational constants are the
            first three elements of each row.
        """
        delta = np.random.choice(shifts, size=(X.shape[0], 3), replace=True, p=p)
        shift = 100.0 - delta
        X[:, :3] = (100.0 * X[:, :3]) / shift

    @jit
    def __remove_blanks__(self, array: np.ndarray):
        mask = np.argwhere(array.sum(axis=[1, 2]) != 0.0)
        return mask, array[mask]

    def __data_generation__(self, indexes: np.ndarray):
        """
        Function that retrieves NumPy ndarrays from an HDF5 file.
        X, Y correspond to the data and labels respectively.
        """
        indexes = np.sort(indexes)
        X = self.h5_obj[self.X_label][indexes]
        Y = self.h5_obj[self.Y_label][indexes]
        # Reshape the data if it's 2D, and we'll assume it's a square
        # matrix
        # if Y.ndim == 3:
        #     Y = np.reshape(Y, (Y.shape[0], Y.shape[1] ** 2))
        # if X.ndim == 3:
        #     X = np.reshape(X, (X.shape[0], X.shape[1] ** 2))
        if self.augment is True:
            if X.shape[1] > 8:
                X += np.random.normal(0., self.sigma, X.shape)
            elif X.shape[1] <= 8:
                self.augment_rotational_constants(X, self.__X_shifts, self.__p_shifts)
            else:
                raise Exception("Augmentation specified, but the array shapes are weird!")
        # Whether or not to add a Gaussian blur to the data
        # if self.augment is True:
        #    X = self.__augment_noise__(X, self.sigma)
        return X, Y

    def __getitem__(self, index: int):
        """
        Method that yields the data to Keras function calls.
        """
        # Determine the range of index values to extract
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        X, Y = self.__data_generation__(indexes)
        return X, Y
