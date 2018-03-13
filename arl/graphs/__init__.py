"""Modules for graph processing using dask

"""
from arl.data.parameters import set_parameters


def graphs_default_args(arl_config='arl_config.ini'):
    d = {'nchan': 1,
         'algorithm': 'mmclean',
    }
    
    set_parameters(arl_config, d, 'graphs')
