"""
Functions that perform fourier transform processing, both Image->Visibility (predict) and Visibility->Image (
invert). In  addition there are functions for predicting visibilities from Skycomponents.

For example::

    model = create_image_from_visibility(vt, cellsize=0.001, npixel=256)
    dirty, sumwt = invert_function(vt, model, context='2d')
    psf, sumwt = invert_function(vt, model, dopsf=True, context='2d')

The principal transitions between the data models are:

.. image:: ./ARL_transitions.png
   :scale: 75 %

The parameters active in this module are::

    [imaging]
    facets = 1
    padding = 2
    oversampling = 8
    wstep = 0.0
    wstack = 4.0
    vis_slices = 10
    timeslice = auto
    
"""

from arl.imaging.params import get_polarisation_map, get_rowmap, get_uvw_map, standard_kernel_list, \
    w_kernel_list, get_kernel_list, advise_wide_field
from arl.imaging.base import predict_2d, predict_skycomponent_visibility, \
    predict_skycomponent_visibility, invert_2d, normalize_sumwt, shift_vis_to_image, \
    create_image_from_visibility, residual_image
from arl.imaging.imaging_context import imaging_context, invert_function, predict_function

from arl.data.parameters import set_parameters


def imaging_default_args(arl_config='arl_config.ini'):
    d = {
        'facets': 1,
        'padding': 2,
        'oversampling': 8,
        'wstep': 0.0,
        'wstack': 0.0,
        'vis_slices': 1,
        'timeslice': 'auto'}
    
    set_parameters(arl_config, d, 'imaging')
