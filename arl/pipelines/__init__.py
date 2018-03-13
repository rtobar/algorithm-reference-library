""" Pipeline functions

"""
from arl.data.parameters import set_parameters


def pipelines_default_args(arl_config='arl_config.ini'):
    d = {
        'nmajor': 5,
        'do_selfcal': True,
        'threshold': 0.0
    }
    
    set_parameters(arl_config, d, 'pipelines')

