from __future__ import division
import scipy.io as spio
import numpy as np
import os

def get_HGdata(filename, var_list):
    """
    reads a .mat file as a dictionary and returns requested variables
    input:
        filename = full path to data (saved as .mat file)
        var_list = list of variables names saved in .mat file.
                For example data, possible values are:
                onsets_stim, onsets_resp = onset time per trial
                data = HG data matrix - elecs x trials x time
                srate = sampling rate
                active_elecs = electrode indices
    output:
        list of variable values from var_list
    """
    
    data_dict = loadmat(filename) #load the data dictionary
    variables = [data_dict[k] for k in var_list]
    return variables


def loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    see: http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    """
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
