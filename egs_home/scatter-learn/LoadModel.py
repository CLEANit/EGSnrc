#!/home/user/miniconda3/envs/skmod/bin/python
import numpy as np
import pickle
import pandas as pd
import sys

#from ProcessInput import ProcessInput

def GetExpandedShape(input_data):
    input_data_shape = input_data.shape
    input_data_shape = [1] + list(input_data_shape)
    input_data_shape = tuple(input_data_shape)
    return input_data_shape

def InferNPad(mod):
    from numpy import sqrt
    try:
        npad = mod.n_features_in_
    except AttributeError:
        npad = mod.chorus.estimators_[0].n_features_in_

    npad = round(sqrt(npad) - 1)
    return npad

from pickle import load
def LoadModel(mod_path="trained_model.pkl"):
    with open(mod_path,'rb') as mf:
            mod = load(mf)
    return mod

