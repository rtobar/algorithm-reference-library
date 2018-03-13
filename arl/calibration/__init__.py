""" Module for calibration

"""
from arl.data.parameters import set_parameters


def calibration_default_args(arl_config='arl_config.ini'):
    
    d = {
        'gt_slices': 1,
        'timeslice': 'auto',
        'frequencyslice': 'auto',
        'gain': 0.1,
        'niter': 3
    }
    
    set_parameters(arl_config, d, 'calibration')
