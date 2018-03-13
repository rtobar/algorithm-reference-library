""" Functions for processing visibility. These operate on one or both of BlockVisibility and Visibility.

"""

from arl.data.parameters import set_parameters


def visibility_default_args(arl_config='arl_config.ini'):
    d = {
        }
    
    set_parameters(arl_config, d, 'visibility')

