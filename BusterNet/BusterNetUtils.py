import numpy as np
from matplotlib import pyplot


def simple_cmfd_decoder(busterNetModel, rgb):
    """A simple BusterNet CMFD decoder"""
    # 1. expand an image to a single sample batch
    single_sample_batch = np.expand_dims(rgb, axis=0)
    # 2. perform busterNet CMFD
    pred = busterNetModel.predict(single_sample_batch)[0]
    return pred

