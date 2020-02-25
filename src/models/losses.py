import tensorflow as tf
from tensorflow.keras import backend as K


def nonzero_mean_squared_error(y_true, y_pred):
    """
    Loss function that ignores zero-padding. This works
    by using tensor masking, such that when it comes to
    computing the mean squared error we are ignoring all
    of the zeros during the calculation.
    
    Parameters
    ----------
    y_true : [type]
        [description]
    y_pred : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    # Mask the arrays
    y_true = y_true * K.cast(y_true > 0, y_pred.dtype)
    y_pred = y_pred * K.cast(y_pred > 0, y_pred.dtype)
    return K.mean(K.square(y_pred - y_true), axis=-1)


def asymmetric_loss(y_true, y_pred, a=0.4):
    """
    A modified version of the mean squared error, which penalizes
    performance asymmetrically with a tunable parameter a ~ [-1,1].
    
    For positive values of a, overestimation is penalized, and
    vice versa.
    
    Parameters
    ----------
    y_true : [type]
        [description]
    y_pred : [type]
        [description]
    a : float, optional
        [description], by default 0.4
    
    Returns
    -------
    [type]
        [description]
    """
    error = y_pred - y_true
    return tf.pow(error, 2) * tf.pow(tf.sign(error) + 0.4, 2)


def quantile_loss(y_true, y_pred, quantile=0.5):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    # Mask the arrays, and remove all zero-padding
    y_true = y_true * K.cast(y_true > 0, y_pred.dtype)
    y_pred = y_pred * K.cast(y_pred > 0, y_pred.dtype)
    # calculate the raw difference
    error = y_pred - y_true
    # This returns the quantile regression loss
    return K.mean(K.maximum(quantile * error, (quantile - 1 * error)), axis=-1)
