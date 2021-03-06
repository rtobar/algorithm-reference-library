"""
Functions to create primary beam modelsw
"""

import collections
import logging
import warnings

import numpy
from astropy import constants as const
from astropy.wcs import FITSFixedWarning
from astropy.wcs.utils import skycoord_to_pixel

from processing_components.image.operations import create_empty_image_like
from processing_components.simulation.testing_support import create_low_test_beam

log = logging.getLogger(__name__)


def ft_disk(r):
    from scipy.special import jn  # pylint: disable=no-name-in-module
    result = numpy.zeros_like(r)
    result[r > 0] = 2.0 * jn(1, r[r > 0]) / r[r > 0]
    rsmall = 1e-9
    result[r == 0] = 2.0 * jn(1, rsmall) / rsmall
    return result


def create_pb(model, telescope='MID', pointingcentre=None):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model: Template image
    :param telescope: 'VLA' or 'ASKAP'
    :return: Primary beam image
    """
    if telescope[0:3] == 'MID':
        return create_pb_generic(model, pointingcentre=pointingcentre, diameter=15.0, blockage=0.0)
    elif telescope[0:3] == 'LOW':
            return create_low_test_beam(model)
    elif telescope[0:3] == 'VLA':
        return create_pb_generic(model, pointingcentre=pointingcentre, diameter=25.0, blockage=1.8)
    elif telescope[0:5] == 'ASKAP':
        return create_pb_generic(model, pointingcentre=pointingcentre, diameter=12.0, blockage=1.0)
    else:
        raise NotImplementedError('Telescope %s has no primary beam model' % telescope)


def mosaic_pb(model, telescope, pointingcentres):
    """ Create a mosaic primary beam by adding primary beams for a set of pointing centres
    
    Note that the addition is root sum of squares
    
    :param model:  Template image
    :param telescope:
    :param pointingcentres:  list of pointing centres
    :return:
    """
    assert isinstance(pointingcentres, collections.Iterable), "Need a list of pointing centres"
    sumpb = create_empty_image_like(model)
    for pc in pointingcentres:
        pb = create_pb(model, telescope, pointingcentre=pc)
        sumpb.data += pb.data ** 2
    sumpb.data = numpy.sqrt(sumpb.data)
    return sumpb


def create_pb_generic(model, pointingcentre=None, diameter=25.0, blockage=1.8):
    """
    Make an image like model and fill it with an analytical model of the primary beam
    :param model:
    :return:
    """
    beam = create_empty_image_like(model)
    
    nchan, npol, ny, nx = model.shape
    
    if pointingcentre is not None:
        cx, cy = skycoord_to_pixel(pointingcentre, model.wcs, 0, 'wcs')
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            cx, cy = beam.wcs.sub(2).wcs.crpix[0] - 1, beam.wcs.sub(2).wcs.crpix[1] - 1
    
    for chan in range(nchan):
        
        # The frequency axis is the second to last in the beam
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FITSFixedWarning)
            frequency = model.wcs.sub(['spectral']).wcs_pix2world([chan], 0)[0]
        wavelength = const.c.to('m s^-1').value / frequency
        
        d2r = numpy.pi / 180.0
        scale = d2r * numpy.abs(beam.wcs.sub(2).wcs.cdelt[0])
        xx, yy = numpy.meshgrid(scale * (range(nx) - cx), scale * (range(ny) - cy))
        # Radius of each cell in radians
        rr = numpy.sqrt(xx ** 2 + yy ** 2)
        
        blockage_factor = (blockage / diameter) ** 2
        
        for pol in range(npol):
            reflector = ft_disk(rr * numpy.pi * diameter / wavelength)
            blockage = ft_disk(rr * numpy.pi * blockage / wavelength)
            beam.data[chan, pol, ...] = reflector - blockage_factor * blockage
    
    beam.data *= beam.data
    return beam
