""" Pipeline functions. SDP standard pipelinee expressed as functions. This is quite slow and is provided mainly for
completeness. Use parallel versions pipelines/graphs.py for speed.

"""
import collections
import logging

import numpy

from arl.calibration.calibration_control import calibrate_function, create_calibration_controls
from arl.data.data_models import Image, BlockVisibility, GainTable
from arl.data.parameters import get_parameter
from arl.image.deconvolution import deconvolve_cube, restore_cube
from arl.imaging import predict_skycomponent_visibility
from arl.imaging.imaging_context import predict_function, invert_function
from arl.visibility.base import copy_visibility
from arl.visibility.coalesce import convert_blockvisibility_to_visibility

log = logging.getLogger(__name__)

def ical(block_vis: BlockVisibility, model: Image, components=None, context='2d', controls=None,
         arl_config='arl_config.ini'):
    """ Post observation image, deconvolve, and self-calibrate
   
    :param vis:
    :param model: Model image
    :param components: Initial components
    :param context: Imaging context
    :param controls: Calibration controls dictionary
    :return: model, residual, restored
    """
    nmajor = int(get_parameter(arl_config, 'nmajor', 5, 'pipelines'))
    log.info("ical: Performing %d major cycles" % nmajor)
    
    do_selfcal = bool(get_parameter(arl_config, "do_selfcal", False, 'pipelines'))

    if controls is None:
        controls = create_calibration_controls(arl_config)
    
    # The model is added to each major cycle and then the visibilities are
    # calculated from the full model
    vis = convert_blockvisibility_to_visibility(block_vis)
    block_vispred = copy_visibility(block_vis, zero=True)
    vispred = convert_blockvisibility_to_visibility(block_vispred)
    vispred.data['vis'][...] = 0.0
    visres = copy_visibility(vispred)
    
    vispred = predict_function(vispred, model, context=context, arl_config=arl_config)
    
    if components is not None:
        vispred = predict_skycomponent_visibility(vispred, components)
    
    if do_selfcal:
        vis, gaintables = calibrate_function(vis, vispred, 'TGB', controls, iteration=-1)
    
    visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
    dirty, sumwt = invert_function(visres, model, context=context, arl_config=arl_config)
    log.info("Maximum in residual image is %.6f" % (numpy.max(numpy.abs(dirty.data))))
    
    psf, sumwt = invert_function(visres, model, dopsf=True, context=context, arl_config=arl_config)
    
    threshold = float(get_parameter(arl_config, "threshold", 0.0, section='pipelines'))
    
    for i in range(nmajor):
        log.info("ical: Start of major cycle %d of %d" % (i, nmajor))
        cc, res = deconvolve_cube(dirty, psf, arl_config=arl_config)
        model.data += cc.data
        vispred.data['vis'][...] = 0.0
        vispred = predict_function(vispred, model, context=context, arl_config=arl_config)
        if do_selfcal:
            vis, gaintables = calibrate_function(vis, vispred, 'TGB', controls, iteration=i)
        visres.data['vis'] = vis.data['vis'] - vispred.data['vis']
        
        dirty, sumwt = invert_function(visres, model, context=context, arl_config=arl_config)
        log.info("Maximum in residual image is %s" % (numpy.max(numpy.abs(dirty.data))))
        if numpy.abs(dirty.data).max() < 1.1 * threshold:
            log.info("ical: Reached stopping threshold %.6f Jy" % threshold)
            break
        log.info("ical: End of major cycle")
    
    log.info("ical: End of major cycles")
    restored = restore_cube(model, psf, dirty, arl_config=arl_config)
    
    return model, dirty, restored


def continuum_imaging(vis: BlockVisibility, model: Image, components=None, context='2d',
                      arl_config='arl_config.ini') -> (Image, Image, Image):
    """Continuum imaging from calibrated (DDE and DIE) and coalesced data

    The model image is used as the starting point, and also to determine the imagesize and sampling. Components
    are subtracted before deconvolution.
    
    Uses :py:func:`arl.image.solvers.solve_image`
    
    :param vis: BlockVisibility
    :param model: model image
    :param components: Component-based sky model
    :param arl_config: Parameters
    :return:
    """
    return ical(vis, model, components=components, context=context, do_selfcal=False, arl_config=arl_config)


def spectral_line_imaging(vis: BlockVisibility, model: Image, continuum_model: Image = None, continuum_components=None,
                          context='2d', arl_config='arl_config.ini') -> (Image, Image, Image):
    """Spectral line imaging from calibrated (DIE) data
    
    A continuum model can be subtracted, and the residual image deconvolved.
    
    :param vis: Visibility
    :param model: Image specify details of model
    :param continuum_model: model continuum image to be subtracted
    :param continuum_components: mode components to be subtracted
    :param spectral_model: model spectral image
    :param predict: Predict fumction e.g. predict_2d
    :param invert: Invert function e.g. invert_wprojection
    :return: Residual visibility, spectral model image, spectral residual image
    """
    
    vis_no_continuum = copy_visibility(vis)
    if continuum_model is not None:
        vis_no_continuum = predict_function(vis_no_continuum, continuum_model, context=context, arl_config=arl_config)
    if continuum_components is not None:
        vis_no_continuum = predict_skycomponent_visibility(vis_no_continuum, continuum_components)
    vis_no_continuum.data['vis'] = vis.data['vis'] - vis_no_continuum.data['vis']
    
    log.info("spectral_line_imaging: Deconvolving continuum subtracted visibility")
    return ical(vis_no_continuum.data, model, components=None, context=context, do_selfcal=False, arl_config=arl_config)


def fast_imaging(arl_config) -> (Image, Image, Image):
    """Fast imaging from calibrated (DIE only) data

    :param arl_config: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True


def eor(arl_config) -> (Image, Image, Image):
    """eor calibration and imaging
    
    :param arl_config: Dictionary containing parameters
    :return:
    """
    # TODO: implement
    
    return True


def rcal(vis: BlockVisibility, components, arl_config='arl_config.ini') -> GainTable:
    """ Real-time calibration pipeline.

    Reads visibilities through a BlockVisibility iterator, calculates model visibilities according to a
    component-based sky model, and performs calibration solution, writing a gaintable for each chunk of
    visibilities.

    :param vis: Visibility or Union(Visibility, Iterable)
    :param components: Component-based sky model
    :param arl_config: Parameters
    :return: gaintable
   """
    
    if not isinstance(vis, collections.Iterable):
        vis = [vis]
    
    from arl.calibration.solvers import solve_gaintable
    for ichunk, vischunk in enumerate(vis):
        vispred = copy_visibility(vischunk, zero=True)
        vispred = predict_skycomponent_visibility(vispred, components)
        gt = solve_gaintable(vischunk, vispred, arl_config=arl_config)
        yield gt
