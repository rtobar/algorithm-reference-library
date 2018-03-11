#
"""
Definition of structures needed by the function interface. These are mostly
subclasses of astropy classes.
"""

import logging

import numpy

from arl.data.data_models import Visibility, Image
from arl.image.deconvolution import deconvolve_cube
from arl.imaging.base import predict_skycomponent_visibility
from arl.imaging.imaging_context import predict_function, invert_function
from arl.visibility.base import copy_visibility

log = logging.getLogger(__name__)


def solve_image(vis: Visibility, model: Image, components=None, context='2d', nmajor=5,
                niter=1000, threshold=0.01, gain=0.7, psf_support=200,
                window='quarter', scales=[0, 3, 10, 30],
                fractional_threshold=0.1, algorithm='msclean',
                arl_config='arl_config.ini') -> (Visibility, Image, Image):
    """Solve for image using deconvolve_cube and specified predict, invert

    This is the same as a majorcycle/minorcycle algorithm. The components are removed prior to deconvolution.
    
    See also arguments for predict, invert, deconvolve_cube functions.2d

    :param vis:
    :param model: Model image
    :param predict: Predict function e.g. predict_2d, predict_wstack
    :param invert: Invert function e.g. invert_2d, invert_wstack
    :return: Visibility, model
    """
    log.info("solve_image: Performing %d major cycles" % nmajor)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vispred = copy_visibility(vis)
    visres = copy_visibility(vis)
    
    vispred = predict_function(vispred, model, context=context)
    
    if components is not None:
        vispred = predict_skycomponent_visibility(vispred, components)
    
    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert_function(visres, model, context=context, dopsf=False, arl_config=arl_config)
    psf, sumwt = invert_function(visres, model, context=context, dopsf=True, arl_config=arl_config)
    
    for i in range(nmajor):
        log.info("solve_image: Start of major cycle %d" % i)
        cc, res = deconvolve_cube(dirty, psf, niter=niter, threshold=threshold, gain=gain,
                                  psf_support=psf_support, window=window, scales=scales,
                                  fractional_threshold=fractional_threshold,
                                  algorithm=algorithm)
        model.data += cc.data
        vispred = predict_function(vispred, model, context=context, arl_config=arl_config)
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        dirty, sumwt = invert_function(visres, model, context=context, dopsf=False, arl_config=arl_config)
        if numpy.abs(dirty.data).max() < 1.1 * threshold:
            log.info("Reached stopping threshold %.6f Jy" % threshold)
            break
        log.info("solve_image: End of major cycle")
    
    log.info("solve_image: End of major cycles")
    return visres, model, dirty
