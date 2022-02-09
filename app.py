import os
from matplotlib import pyplot as plt
from MantraNet.mantranet import pre_trained_model, check_forgery
import streamlit as st

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

st.header("IMD Demo")
device = "cpu"  # to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
MantraNetmodel = pre_trained_model(
    weight_path="MantraNet/MantraNetv4.pt", device=device
)
# busterNetModel = create_BusterNet_testing_model( 'BusterNet/pretrained_busterNet.hd5' )


def check_image(img_path):
    device = "cpu"  # to change if you have a GPU with at least 12Go RAM (it will save you a lot of time !)
    MantraNetmodel = pre_trained_model(
        weight_path="MantraNet/MantraNetv4.pt", device=device
    )
    fig = check_forgery(MantraNetmodel, img_path=img_path, device=device)
    st.pyplot(fig)
