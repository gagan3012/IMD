"""
This file defines all BusterNet related custom layers
"""
from __future__ import print_function
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Layer, Input, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import backend as K
import tensorflow as tf


def std_norm_along_chs(x):
    """Data normalization along the channle axis
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        xn = tensor4d, same shape as x, normalized version of x
    """
    avg = K.mean(x, axis=-1, keepdims=True)
    std = K.maximum(1e-4, K.std(x, axis=-1, keepdims=True))
    return (x - avg) / std


def BnInception(x, nb_inc=16, inc_filt_list=[(1, 1), (3, 3), (5, 5)], name="uinc"):
    """Basic Google inception module with batch normalization
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
        nb_inc = int, number of filters in individual Conv2D
        inc_filt_list = list of kernel sizes, individual Conv2D kernel size
        name = str, name of module
    Output:
        xn = tensor4d, (n_samples, n_rows, n_cols, n_new_feats)
    """
    uc_list = []
    for idx, ftuple in enumerate(inc_filt_list):
        uc = Conv2D(
            nb_inc,
            ftuple,
            activation="linear",
            padding="same",
            name=name + "_c%d" % idx,
        )(x)
        uc_list.append(uc)
    if len(uc_list) > 1:
        uc_merge = Concatenate(axis=-1, name=name + "_merge")(uc_list)
    else:
        uc_merge = uc_list[0]
    uc_norm = BatchNormalization(name=name + "_bn")(uc_merge)
    xn = Activation("relu", name=name + "_re")(uc_norm)
    return xn


class SelfCorrelationPercPooling(Layer):
    """Custom Self-Correlation Percentile Pooling Layer
    Arugment:
        nb_pools = int, number of percentile poolings
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x_pool = tensor4d, (n_samples, n_rows, n_cols, nb_pools)
    """

    def __init__(self, nb_pools=256, **kwargs):
        self.nb_pools = nb_pools
        super(SelfCorrelationPercPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask=None):
        # parse input feature shape
        bsize, nb_rows, nb_cols, nb_feats = K.int_shape(x)
        nb_maps = nb_rows * nb_cols
        # self correlation
        x_3d = K.reshape(x, tf.stack([-1, nb_maps, nb_feats]))
