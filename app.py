import os
from matplotlib import pyplot as plt
from MantraNet.mantranet import pre_trained_model, check_forgery
import streamlit as st

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.header("IMD Demo")
device = "cpu"  # to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
