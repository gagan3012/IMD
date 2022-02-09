import os
from matplotlib import pyplot as plt
from MantraNet.mantranet import pre_trained_model, check_forgery
import streamlit as st

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

