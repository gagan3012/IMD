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
        x_corr_3d = (
            tf.matmul(x_3d, x_3d, transpose_a=False, transpose_b=True) / nb_feats
        )
        x_corr = K.reshape(x_corr_3d, tf.stack([-1, nb_rows, nb_cols, nb_maps]))
        # argsort response maps along the translaton dimension
        if self.nb_pools is not None:
            ranks = K.cast(
                K.round(tf.linspace(1.0, nb_maps - 1, self.nb_pools)), "int32"
            )
        else:
            ranks = tf.range(1, nb_maps, dtype="int32")
        x_sort, _ = tf.nn.top_k(x_corr, k=nb_maps, sorted=True)
        # pool out x features at interested ranks
        # NOTE: tf v1.1 only support indexing at the 1st dimension
        x_f1st_sort = K.permute_dimensions(x_sort, (3, 0, 1, 2))
        x_f1st_pool = tf.gather(x_f1st_sort, ranks)
        x_pool = K.permute_dimensions(x_f1st_pool, (1, 2, 3, 0))
        return x_pool

    def compute_output_shape(self, input_shape):
        bsize, nb_rows, nb_cols, nb_feats = input_shape
        nb_pools = (
            self.nb_pools if (self.nb_pools is not None) else (nb_rows * nb_cols - 1)
        )
        return tuple([bsize, nb_rows, nb_cols, nb_pools])


class BilinearUpSampling2D(Layer):
    """Custom 2x bilinear upsampling layer
    Input:
        x = tensor4d, (n_samples, n_rows, n_cols, n_feats)
    Output:
        x2 = tensor4d, (n_samples, 2*n_rows, 2*n_cols, n_feats)
    """

    def call(self, x, mask=None):
        bsize, nb_rows, nb_cols, nb_filts = K.int_shape(x)
        new_size = tf.constant([nb_rows * 2, nb_cols * 2], dtype=tf.int32)
        return tf.image.resize(x, new_size)

    def compute_output_shape(self, input_shape):
        bsize, nb_rows, nb_cols, nb_filts = input_shape
        return tuple([bsize, nb_rows * 2, nb_cols * 2, nb_filts])


class ResizeBack(Layer):
    """Custom bilinear resize layer
    Resize x's spatial dimension to that of r

    Input:
        x = tensor4d, (n_samples, n_rowsX, n_colsX, n_featsX )
        r = tensor4d, (n_samples, n_rowsR, n_colsR, n_featsR )
    Output:
        xn = tensor4d, (n_samples, n_rowsR, n_colsR, n_featsX )
    """

    def call(self, x):
        t, r = x
        new_size = [tf.shape(r)[1], tf.shape(r)[2]]
        return tf.image.resize(t, new_size)

    def compute_output_shape(self, input_shapes):
        tshape, rshape = input_shapes
        return (tshape[0],) + rshape[1:3] + (tshape[-1],)


class Preprocess(Layer):
    """Basic preprocess layer for BusterNet

    More precisely, it does the following two things
    1) normalize input image size to (256,256) to speed up processing
    2) substract channel-wise means if necessary
    """

    def call(self, x, mask=None):
        # parse input image shape
        bsize, nb_rows, nb_cols, nb_colors = K.int_shape(x)
        if (nb_rows != 256) or (nb_cols != 256):
            # resize image if different from (256,256)
            x256 = tf.image.resize(x, [256, 256], name="resize")
        else:
            x256 = x
        # substract channel means if necessary
        if K.dtype(x) == "float32":
            # input is not a 'uint8' image
            # assume it has already been normalized
            xout = x256
        else:
            # input is a 'uint8' image
            # substract channel-wise means
            xout = preprocess_input(x256)
        return xout

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 256, 256, 3)


def create_cmfd_similarity_branch(
    img_shape=(256, 256, 3), nb_pools=100, name="simiDet"
):
    """Create the similarity branch for copy-move forgery detection"""
    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------
    img_input = Input(shape=img_shape, name=name + "_in")
    # ---------------------------------------------------------
    # VGG16 Conv Featex
    # ---------------------------------------------------------
    bname = name + "_cnn"
    ## Block 1
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same", name=bname + "_b1c1")(
        img_input
    )
    x1 = Conv2D(64, (3, 3), activation="relu", padding="same", name=bname + "_b1c2")(x1)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name=bname + "_b1p")(x1)
    # Block 2
    x2 = Conv2D(128, (3, 3), activation="relu", padding="same", name=bname + "_b2c1")(
        x1
    )
    x2 = Conv2D(128, (3, 3), activation="relu", padding="same", name=bname + "_b2c2")(
        x2
