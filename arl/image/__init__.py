""" Module for image operations, including deconvolution

The parameters active in this module are:

[deconvolution]
niter = 10000
gain = 0.1
algorithm = mmclean
scales = [0]
threshold = 0.01
nmoments = 1
findpeak = ARL
window = quarter
nmajor = 1

"""
from arl.data.parameters import set_parameters

def image_default_args(arl_config='arl_config.ini'):
    d = {'psf_support': 200,
         'window': 'quarter',
         'niter': 10000,
         'gain': 0.1,
         'algorithm': 'hogbom',
         'threshold': 0.01,
         'fractional_threshold': 0.1,
         'nmoments': 1,
         'scales': [0, 3, 10, 30],
         'findpeak': 'ARL',
         'return_moments': False,
         'window_shape': ''}
    
    set_parameters(arl_config, d, 'deconvolution')
