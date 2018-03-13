""" Image deconvolution functions

The standard deconvolution algorithms are provided:

    hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)
    
    msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    mfsmsclean: MultiScale Multi-Frequency See: U. Rau and T. J. Cornwell, “A multi-scale multi-frequency
    deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532, A71 (2011).

For example to make dirty image and PSF, deconvolve, and then restore::

    model = create_image_from_visibility(vt, cellsize=0.001, npixel=256)
    dirty, sumwt = invert_2d(vt, model)
    psf, sumwt = invert_2d(vt, model, dopsf=True)

    comp, residual = deconvolve_cube(dirty, psf, niter=1000, threshold=0.001, fracthresh=0.01, window='quarter',
                                 gain=0.7, algorithm='msclean', scales=[0, 3, 10, 30])

    restored = restore_cube(comp, psf, residual)

"""

import logging

import numpy
from astropy.convolution import Gaussian2DKernel, convolve
from photutils import fit_2dgaussian

from arl.data.data_models import Image
from arl.data.parameters import set_parameters, get_parameter
from arl.image.cleaners import hogbom, msclean, msmfsclean
from arl.image.operations import create_image_from_array, copy_image, \
    calculate_image_frequency_moments, calculate_image_from_frequency_moments

log = logging.getLogger(__name__)


def deconvolve_args(arl_config='arl_config.ini'):
    d = {'psf_support': 200, 'window': 'quarter', 'niter': 10000, 'gain': 0.1,
         'algorithm': 'hogbom', 'threshold': 0.01, 'fractional_threshold': 0.1,
         'nmoments': 1, 'scales': [0, 3, 10, 30],
         'findpeak': 'ARL', 'return_moments': False, 'window_shape': ''}
    
    set_parameters(arl_config, d, 'deconvolution')


def deconvolve_cube(dirty: Image, psf: Image, arl_config='arl_config.ini') -> (Image, Image):
    """ Clean using a variety of algorithms
    
    Functions that clean a dirty image using a point spread function. The algorithms available are:
    
    hogbom: Hogbom CLEAN See: Hogbom CLEAN A&A Suppl, 15, 417, (1974)
    
    msclean: MultiScale CLEAN See: Cornwell, T.J., Multiscale CLEAN (IEEE Journal of Selected Topics in Sig Proc,
    2008 vol. 2 pp. 793-801)

    mfsmsclean, msmfsclean, mmclean: MultiScale Multi-Frequency See: U. Rau and T. J. Cornwell,
    “A multi-scale multi-frequency deconvolution algorithm for synthesis imaging in radio interferometry,” A&A 532,
    A71 (2011).
    
    For example::
    
        comp, residual = deconvolve_cube(dirty, psf, niter=1000, gain=0.7, algorithm='msclean',
                                         scales=[0, 3, 10, 30], threshold=0.01)
                                         
    For the MFS clean, the psf must have number of channels >= 2 * nmoments
    
    :param dirty: Image dirty image
    :param psf: Image Point Spread Function
    :param window: Window image (Bool) - clean where True
    :param algorithm: Cleaning algorithm: 'msclean'|'hogbom'|'mfsmsclean'
    :param gain: loop gain (float) 0.7
    :param threshold: Clean threshold (0.0)
    :param fractional_threshold: Fractional threshold (0.01)
    :param scales: Scales (in pixels) for multiscale ([0, 3, 10, 30])
    :param nmoments: Number of frequency moments (default 3)
    :param findpeak: Method of finding peak in mfsclean: 'Algorithm1'|'ASKAPSoft'|'CASA'|'ARL', Default is ARL.
    :return: componentimage, residual
    
    """
    psf_support = int(get_parameter(arl_config, 'psf_support', 200, 'deconvolution'))
    gain = float(get_parameter(arl_config, 'gain', 0.1, 'deconvolution'))
    niter = int(get_parameter(arl_config, 'niter', 1000, 'deconvolution'))
    window = get_parameter(arl_config, 'window', 'quarter', 'deconvolution')
    algorithm = get_parameter(arl_config, 'algorithm', 'hogbom', 'deconvolution')
    threshold = float(get_parameter(arl_config, 'threshold', 0.01, 'deconvolution'))
    fractional_threshold = float(get_parameter(arl_config, 'fractional_threshold', 0.1, 'deconvolution'))
    nmoments = int(get_parameter(arl_config, 'nmoments', 1, 'deconvolution'))
    findpeak = get_parameter(arl_config, 'findpeak', 'ARL', 'deconvolution')
    return_moments = bool(get_parameter(arl_config, 'return_moments', False, 'deconvolution'))
    window_shape = get_parameter(arl_config, 'window_shape', '', 'deconvolution')
    scales = get_parameter(arl_config, 'scales', [0, 3, 10, 30], 'deconvolution')

    assert isinstance(dirty, Image), dirty
    assert isinstance(psf, Image), psf
    
    if window_shape == 'quarter':
        qx = dirty.shape[3] // 4
        qy = dirty.shape[2] // 4
        window = numpy.zeros_like(dirty.data)
        window[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
        log.info('deconvolve_cube: Cleaning inner quarter of each sky plane')
    else:
        window = None
    
    if isinstance(psf_support, int):
        if (psf_support < psf.shape[2] // 2) and ((psf_support < psf.shape[3] // 2)):
            centre = [psf.shape[2] // 2, psf.shape[3] // 2]
            psf.data = psf.data[..., (centre[0] - psf_support):(centre[0] + psf_support),
                       (centre[1] - psf_support):(centre[1] + psf_support)]
            log.info('deconvolve_cube: PSF support = +/- %d pixels' % (psf_support))
    
    if algorithm == 'msclean':
        log.info("deconvolve_cube: Multi-scale clean of each polarisation and channel separately")
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        assert threshold >= 0.0
        assert niter > 0
        assert 0.0 < fractional_threshold < 1.0
        
        comp_array = numpy.zeros_like(dirty.data)
        residual_array = numpy.zeros_like(dirty.data)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.info("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                    None, gain, threshold, niter, scales, fractional_threshold)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            msclean(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                    window[channel, pol, :, :], gain, threshold, niter, scales, fractional_threshold)
                else:
                    log.info("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
        
        comp_image = create_image_from_array(comp_array, dirty.wcs, dirty.polarisation_frame)
        residual_image = create_image_from_array(residual_array, dirty.wcs, dirty.polarisation_frame)
    
    elif algorithm == 'msmfsclean' or algorithm == 'mfsmsclean' or algorithm == 'mmclean':
        
        log.info("deconvolve_cube: Multi-scale multi-frequency clean of each polarisation separately")
        assert nmoments > 0, "Number of frequency moments must be greater than zero"
        nchan = dirty.shape[0]
        assert nchan > 2 * nmoments, "Require nchan %d > 2 * nmoments %d" % (nchan, 2 * nmoments)
        dirty_taylor = calculate_image_frequency_moments(dirty, nmoments=nmoments)
        psf_taylor = calculate_image_frequency_moments(psf, nmoments=2 * nmoments)
        
        assert 0.0 < gain < 2.0, "Loop gain must be between 0 and 2"
        comp_array = numpy.zeros(dirty_taylor.data.shape)
        residual_array = numpy.zeros(dirty_taylor.data.shape)
        for pol in range(dirty_taylor.data.shape[1]):
            if psf_taylor.data[0, pol, :, :].max():
                log.info("deconvolve_cube: Processing pol %d" % (pol))
                if window is None:
                    comp_array[:, pol, :, :], residual_array[:, pol, :, :] = \
                        msmfsclean(dirty_taylor.data[:, pol, :, :], psf_taylor.data[:, pol, :, :],
                                   None, gain, threshold, niter, scales, fractional_threshold, findpeak)
                else:
                    qx = dirty.shape[3] // 4
                    qy = dirty.shape[2] // 4
                    window_taylor = numpy.zeros_like(dirty_taylor.data)
                    window_taylor[..., (qy + 1):3 * qy, (qx + 1):3 * qx] = 1.0
                    log.info('deconvolve_cube: Cleaning inner quarter of each moment plane')
                    
                    comp_array[:, pol, :, :], residual_array[:, pol, :, :] = \
                        msmfsclean(dirty_taylor.data[:, pol, :, :], psf_taylor.data[:, pol, :, :],
                                   window_taylor[0, pol, :, :], gain, threshold, niter, scales, fractional_threshold,
                                   findpeak)
            else:
                log.info("deconvolve_cube: Skipping pol %d" % (pol))
        
        comp_image = create_image_from_array(comp_array, dirty_taylor.wcs, dirty.polarisation_frame)
        residual_image = create_image_from_array(residual_array, dirty_taylor.wcs, dirty.polarisation_frame)
        
        if not return_moments:
            log.info("deconvolve_cube: calculating spectral cubes")
            comp_image = calculate_image_from_frequency_moments(dirty, comp_image)
            residual_image = calculate_image_from_frequency_moments(dirty, residual_image)
        else:
            log.info("deconvolve_cube: constructed moment cubes")
    
    elif algorithm == 'hogbom':
        log.info("deconvolve_cube: Hogbom clean of each polarisation and channel separately")
        
        comp_array = numpy.zeros(dirty.data.shape)
        residual_array = numpy.zeros(dirty.data.shape)
        for channel in range(dirty.data.shape[0]):
            for pol in range(dirty.data.shape[1]):
                if psf.data[channel, pol, :, :].max():
                    log.info("deconvolve_cube: Processing pol %d, channel %d" % (pol, channel))
                    if window is None:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                   None, gain, threshold, niter, fractional_threshold)
                    else:
                        comp_array[channel, pol, :, :], residual_array[channel, pol, :, :] = \
                            hogbom(dirty.data[channel, pol, :, :], psf.data[channel, pol, :, :],
                                   window[channel, pol, :, :], gain, threshold, niter, fractional_threshold)
                else:
                    log.info("deconvolve_cube: Skipping pol %d, channel %d" % (pol, channel))
        
        comp_image = create_image_from_array(comp_array, dirty.wcs, dirty.polarisation_frame)
        residual_image = create_image_from_array(residual_array, dirty.wcs, dirty.polarisation_frame)
    else:
        raise ValueError('deconvolve_cube: Unknown algorithm %s' % algorithm)
    
    return comp_image, residual_image


def restore_cube(model: Image, psf: Image, residual=None, psf_width=3.0, arl_config='arl_config.ini') -> Image:
    """ Restore the model image to the residuals

    :params psf: Input PSF
    :return: restored image

    """
    assert isinstance(model, Image), model
    assert isinstance(psf, Image), psf
    assert residual is None or isinstance(residual, Image), residual
    
    restored = copy_image(model)
    
    npixel = psf.data.shape[3]
    sl = slice(npixel // 2 - 7, npixel // 2 + 8)
    
    psf_width = int(get_parameter(arl_config, 'psf_width', 3.0, 'deconvolution'))
    
    if psf_width is None:
        # isotropic at the moment!
        try:
            fit = fit_2dgaussian(psf.data[0, 0, sl, sl])
            if fit.x_stddev <= 0.0 or fit.y_stddev <= 0.0:
                log.debug('restore_cube: error in fitting to psf, using 1 pixel stddev')
                psf_width = 1.0
            else:
                psf_width = max(fit.x_stddev, fit.y_stddev)
                log.debug('restore_cube: psfwidth = %s' % (psf_width))
        except ValueError as err:
            log.debug('restore_cube: warning in fit to psf, using 1 pixel stddev')
            psf_width = 1.0
    else:
        log.debug('restore_cube: Using specified psfwidth = %s' % (psf_width))
    
    # By convention, we normalise the peak not the integral so this is the volume of the Gaussian
    norm = 2.0 * numpy.pi * psf_width ** 2
    gk = Gaussian2DKernel(psf_width)
    for chan in range(model.shape[0]):
        for pol in range(model.shape[1]):
            restored.data[chan, pol, :, :] = norm * convolve(model.data[chan, pol, :, :], gk, normalize_kernel=False)
    if residual is not None:
        restored.data += residual.data
    return restored
