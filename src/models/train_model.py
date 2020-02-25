"""
train_model.py

A series of helper functions and classes to train neural networks
with Keras.
"""

from pathlib import Path
import os
import uuid

from src.models import predict_model

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LeakyReLU,
    BatchNormalization,
    Reshape,
    LSTM,
    TimeDistributed,
    Bidirectional
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow import keras


class KerasModel(Model):
    def __init__(self, layers, name=None, training=True, **kwargs):
        # This model ID is for bookkeeping - no two models are the same.
        if name is None:
            name = str(uuid.uuid4().hex[:6])
        else:
            assert type(name) == str
        inputs, outputs = self.build_model(layers)
        super().__init__(name=name, inputs=inputs, outputs=outputs, **kwargs)
        # Same folders for organizing
        for folder in ["models", "json", "tensorboard"]:
            if os.path.exists(folder) is False:
                os.mkdir(folder)
        # Initialize the tensorboard callback
        self.tb_callback = TensorBoard(
            log_dir=f"tensorboard/{self.name}",
            histogram_freq=1,
            write_graph=False,
            write_images=True,
        )
        # For performing dropout predictions, this forces the
        # dropout layers to act even during prediction phase
        if training is True:
            K.set_learning_phase(True)

    def build_model(self, layers):
        """
        Helper function to build a Keras Model object; this is
        primarily written to support dropouts while performing
        predictions.
        
        Parameters
        ----------
        compile : bool, optional
            Whether or not to compile the model after building the layers, by default True
        dropout_predict : bool, optional
            Whether or not to use dropouts during the prediction phase, by default True
        kwargs :
            Additional kwargs are passed into the `compile` function call.
        """
        input_layer = layers[0]
        network = input_layer
        # Ignore the first element, which is the input layer
        for layer in layers[1:]:
            # If we're using dropout layers and we want to do the dropout predictions
            # this case statement will ensure the dropout layers have the `trainable`
            # boolean set to True
            network = layer(network)
        # Finalize the model
        inputs = [input_layer]
        outputs = network
        return inputs, outputs

    def dropout_predict(self, data: np.ndarray, niter=10):
        """
        Special version of the class method `predict`, where in this case
        a prediction is made with dropouts. This function simply runs
        predictions for a specified number of times, and calculates the
        mean and standard deviation, thereby giving an indication of the
        uncertainty for predictions.
        
        Parameters
        ----------
        data : np.ndarray
            [description]
        niter : int, optional
            [description], by default 10
        
        Returns
        -------
        [type]
            [description]
        """
        n_items = data.shape[0]
        n_outputs = self.output.shape[-1]
        full_predict = np.zeros((niter, n_items, n_outputs))
        for i in range(niter):
            full_predict[i] = self.predict(data)
        mean = np.mean(full_predict, axis=0)
        std = np.std(full_predict, axis=0)
        return mean, std, full_predict

    def dump_config(self, filepath=None):
        """
        Serialize a model into a JSON file. This is more for the
        sake of being able to keep track of possible configurations.
        
        Parameters
        ----------
        filepath : str or None, optional
            If None, uses the `name` and defaults to saving it
            into the json/ folder. If a str is given, save it to the
            specified path.
        """
        if filepath is None:
            filepath = Path(f"json/{self.name}.json")
        else:
            filepath = Path(filepath)
        if os.path.exists(filepath.parent) is False:
            os.mkdir("json")
        with open(filepath, "w+") as write_file:
            json.dump(self.get_config(), write_file, indent=4, sort_keys=True)

    def fit_save(self, json_file=None, **kwargs):
        """
        Specialized method for automating some bookkeeping stuff after training
        and given network. This will dump the network configuration into a JSON
        file, as well as save the model to a HDF5 file.
        
        Parameters
        ----------
        json_file : str or None, optional
            If None, uses the `name` and defaults to saving it
            into the json/ folder. If a str is given, save it to the
            specified path.
        
        Returns
        -------
        history : dict
            Dictionary containing the training history data.
        """
        history = self.fit(callbacks=[self.tb_callback], **kwargs)
        self.dump_config(json_file)
        self.save(self.h5_model, save_format="h5")
        return history

    def _model_gradient(self, inputs, targets, loss_func):
        """
        Function to evaluate the gradient of the model based on a
        set of inputs
        
        Parameters
        ----------
        inputs : [type]
            [description]
        targets : [type]
            [description]
        loss_func : [type]
            [description]
        
        Returns
        -------
        loss_value, gradient : Tensor objects
            [description]
        """
        with tf.GradientTape() as tape:
            # Evaluate the model based on the inputs
            predict = self(inputs)
            loss_value = loss_func(y_true=targets, y_pred=predict)
            gradient = tape.gradient(loss_value, self.trainable_variables)
        return loss_value, gradient

    def _fit_iteration(self, inputs, targets, loss_func=None):
        if loss_func is None:
            loss_func = self.loss
        loss, gradient = self._model_gradient(inputs, targets, loss_func)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

    def set_traininable_dropout(self, dropout_predict=True):
        """
        Function to toggle whether or not dropouts is performed during
        the prediction phase.
        
        Parameters
        ----------
        dropout_predict : bool, optional
            Whether or not to use dropouts in predictions, by default True
        """
        for layer in self.layers:
            if "dropout" in layer.name:
                layer.training = dropout_predict


def tf_sliding_window(array: tf.Tensor, window_size=5) -> tf.Tensor:
    """
    Convert a 1D Tensor into a 2D sliding Tensor. Taken from this SO answer:
    https://stackoverflow.com/a/43612647
    
    Parameters
    ----------
    array : tf.Tensor
        [description]
    window_size : int
        [description]
    
    Returns
    -------
    tf.Tensor
        [description]
    """
    indexer = tf.range(array.size - window_size + 1)
    return tf.map_fn(lambda i: array[i:i + window_size], indexer, dtype=tf.float64)


def leaky_dense_drop(neurons=16, alpha=0.3, rate=0.4, training=True, **kwargs):
    """
    Create a list for a standard combination of NN layers, comprising
    a fully connected layer, followed by a dropout layer and a LeakyReLU
    activation function.
    
    Additional kwargs are passed into the Dense layer, which will support
    regularization and whatnot
    
    Parameters
    ----------
    neurons : int, optional
        [description], by default 16
    alpha : float, optional
        [description], by default 0.3
    rate : float, optional
        [description], by default 0.4
    
    Returns
    -------
    layers : list
    Three-membered list comprising Dense, Dropout, and LeakyReLU
    """
    default = {
        "kernel_initializer": "glorot_normal",
        "bias_initializer": "glorot_normal",
    }
    default.update(**kwargs)
    layers = [
        Dense(neurons, **default),
        CustomDropout(rate, training=training),
        LeakyReLU(alpha),
    ]
    return layers


class CustomDropout(Dropout):
    """
    Wrapped version of Keras' Dropout layer. This version of the class
    facilitates the Bayesian sampling via repeated predictions with dropout.
    """

    def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape, seed, **kwargs)
        self.training = training

    def call(self, inputs, training=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

            if not training:
                return K.in_train_phase(dropped_inputs, inputs, training=self.training)
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs


def run_test(
    model,
    validation_df,
    validation_data,
    input_label="rotational_constants",
    target_label="pca_eigen",
    npredict=2000,
    combined=True
):
    """
    Function to test a model against some validation data.
    
    Parameters
    ----------
    model : [type]
        [description]
    validation_df : [type]
        [description]
    validation_data : h5py `group`
        H5py data group reference
    selected_formulas : list, optional
        [description], by default ["C6H6", "C4O1H4", "C5N2O2H6"]
    input_label : str, optional
        Label used to access the validation input dataset
    target_label : str, optional
        Label used to access the validation target dataset
    combined : bool, optional
        Whether or not the output dimensions are what is expected; if False,
        then an alternate version of the prediction function is used where
        the output array shape is actually different from the validation data
        and needs to be reshaped and averaged
    
    Returns
    -------
    test_data : dict
        Nested dictionary; first layer has keys corresponding to the formula,
        and the second layer has the test data associated with that formula
    """
    test_data = dict()
    for index, row in validation_df.iterrows():
        input_data = validation_data[input_label][index]
        # This boolean sets which prediction method to use
        if combined is True:
            predict_func = predict_model.dropout_predict_model
        else:
            predict_func = predict_model.split_dropout_predict_model
        output = predict_func(model, input_data, npredict)
        # Populate the dictionary with extra metadata
        output["input"] = input_data
        output["index"] = index
        output["target"] = validation_data[target_label][index]
        # Comparison statistics
        output["mse"] = np.mean(np.square(output["target"] - output["mean"]))
        output["mae"] = np.mean(np.abs(output["target"] - output["mean"]))
        test_data[row["smi"]] = output
    return test_data


class KerasFunctional:
    """
    This class is to help build Keras models with the functional API.
    
    The idea is the specific models that I'll be building specific models
    as subclasses of this class, with pre-programmed architectures once
    they've been properly defined.
    
    The reason for doing this is because subclassing Keras `Model` becomes
    an absolute pain in the butt for deserializing because of the eager
    execution.
    """

    def __init__(self, architecture, name="KerasModel"):
        self.architecture = architecture
        self.name = name

    def build_model(self, input_shape, training=True, compile=True, batch_size=None, **kwargs):
        inputs = keras.Input(input_shape, batch_size=batch_size)
        for index, layer in enumerate(self.architecture):
            if index == 0:
                network = layer(inputs)
            else:
                if ("dropout" in layer.name) or ("lstm" in layer.name):
                    network = layer(network, training=training)
                else:
                    network = layer(network)
        model = Model(inputs=[inputs], outputs=network, name=self.name)
        if compile is True:
            compile_kwargs = {
                "optimizer": Adam(lr=5e-4, amsgrad=True),
                "loss": "mse",
                "metrics": ["mse", "mae"],
            }
            compile_kwargs.update(**kwargs)
            model.compile(**compile_kwargs)
        return model


class EigenFormulaDecoderFunctional(KerasFunctional):
    """
    Helper class for setting up the eigenspectrum
    
    Parameters
    ----------
    KerasFunctional : [type]
        [description]
    """

    def __init__(self, name="EigenFormulaDecoder"):
        initializers = {
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "glorot_normal",
        }
        leaky_relu = LeakyReLU(alpha=0.3)
        architecture = [
            BatchNormalization(),
            Dropout(0.3),
            Dense(
                64, activation=leaky_relu, **initializers
            ),
            Dropout(0.4),
            Dense(
                128, activation=leaky_relu, **initializers
            ),
            Dropout(0.4),
            Dense(
                32, activation=leaky_relu, **initializers
            ),
            Dropout(rate=0.3),
            Dense(16, activation=leaky_relu, **initializers),
            Dropout(rate=0.3),
            Dense(8, activation=leaky_relu, **initializers),
            Dropout(rate=0.3),
            Dense(4, activation="relu", **initializers),
        ]
        super().__init__(architecture, name)


class RotConEncoderFunctional(KerasFunctional):
    """
    Helper class for setting up the eigenspectrum
    
    Parameters
    ----------
    KerasFunctional : [type]
        [description]
    """

    def __init__(self, name="RotConEncoder"):
        initializers = {
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "glorot_normal",
        }
        leaky_relu = LeakyReLU(alpha=0.3)
        architecture = [
            Dropout(0.3),
            Dense(
                16,
                activation=leaky_relu,
                kernel_regularizer=l2(0.1),
                bias_regularizer=l2(0.05),
                **initializers,
            ),
            Dropout(rate=0.3),
            Dense(
                24,
                activation=leaky_relu,
                kernel_regularizer=l2(0.1),
                bias_regularizer=l2(0.05),
                **initializers,
            ),
            Dropout(rate=0.3),
            Dense(
                48,
                activation=leaky_relu,
                **initializers,
            ),
            Dropout(rate=0.3),
            Dense(
                128,
                activation=leaky_relu,
                **initializers,
            ),
            Dropout(rate=0.3),
            # No activation on the output layer
            Dense(
                30,
                activation="relu",
                # kernel_regularizer=l2(0.01),
                # bias_regularizer=l2(0.01),
                **initializers,
            ),
        ]
        super().__init__(architecture, name)


class EigenSMILESDecoderFunctional(KerasFunctional):
    """
    Helper class for setting up the eigenspectrum
    
    Parameters
    ----------
    KerasFunctional : [type]
        [description]
    """

    def __init__(self, name="EigenSMILESDecoder"):
        initializers = {
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "glorot_normal",
        }
        leaky_relu = LeakyReLU(alpha=0.3)
        architecture = [
            Dropout(0.4),
            Dense(
                24, activation=leaky_relu, **initializers
            ),
            Dropout(0.5),
            Dense(
                18, activation=leaky_relu, **initializers
            ),
            Dropout(rate=0.4),
            Dense(30, activation=leaky_relu, **initializers),
            Dropout(rate=0.4),
            Dense(128, activation=leaky_relu, **initializers),
            Dropout(rate=0.4),
            Dense(512, activation=leaky_relu, **initializers),
            Dropout(rate=0.4),
            Dense(1024, activation=leaky_relu, **initializers),
            Dropout(rate=0.4),
            Dense(1500, activation=leaky_relu, **initializers),
            Dropout(rate=0.4),
            Dense(2030, activation="softmax", **initializers),
            Reshape((70, 29))
        ]
        super().__init__(architecture, name)


class EigenFormulaLSTMDecoderFunctional(KerasFunctional):
    """
    Model for the eigenspectrum to SMILES decoder with LSTM
    units. The output here is designed to mimic this tutorial:
    https://www.tensorflow.org/tutorials/text/text_generation
    
    The final Dense layer has 30 units, which corresponds to
    the size of our "vocabulary"; index 0 is blank, each 
    successive element then corresponds to the encoding defined
    in `parse_calculations.onehot_smiles` (which is length 29)
    to give a total of 30 possible outcomes.
    
    The output of which corresponds to the logit (p / 1 - p)
    which is the raw output of a neuron without any activation
    function. If you were to run the output of this tensor
    through a softmax activation, you would get the likelihood
    for each index.
    
    Parameters
    ----------
    KerasFunctional : [type]
        [description]
    """
    def __init__(self, name="EigenSMILESDecoder"):
        initializers = {
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "glorot_normal",
        }
        lstm_settings = {
            "recurrent_dropout": 0.2,
            "dropout": 0.2,
            "kernel_initializer": "glorot_normal",
            "bias_initializer": "glorot_uniform"
        }
        leaky_relu = LeakyReLU(alpha=0.3)
        architecture = [
            BatchNormalization(),
            LSTM(30, return_sequences=True, **lstm_settings),
            LSTM(50, return_sequences=True, **lstm_settings),
            LSTM(100, return_sequences=True, **lstm_settings),
            # LSTM(250, return_sequences=True, **lstm_settings),
            # LSTM(500, return_sequences=True, **lstm_settings),
            TimeDistributed(Dense(200, activation=leaky_relu, **initializers)),
            TimeDistributed(Dense(30, activation="softmax", **initializers)),
        ]
        super().__init__(architecture, name)
