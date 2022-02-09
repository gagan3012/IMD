import numpy as np
from matplotlib import pyplot


def simple_cmfd_decoder(busterNetModel, rgb):
    """A simple BusterNet CMFD decoder"""
    # 1. expand an image to a single sample batch
