"""Manages the imaging context. This take a string and returns a dictionary containing:
 * Predict function
 * Invert function
 * image_iterator function
 * vis_iterator function

"""

import logging

import numpy

from arl.data.data_models import Visibility, Image
from arl.image.iterators import image_raster_iter, image_null_iter
from arl.image.operations import create_empty_image_like
from arl.imaging import normalize_sumwt
from arl.imaging import predict_2d, invert_2d
from arl.imaging.timeslice import predict_timeslice_single, invert_timeslice_single
from arl.imaging.wstack import predict_wstack_single, invert_wstack_single
from arl.visibility.base import copy_visibility, create_visibility_from_rows
from arl.visibility.coalesce import coalesce_visibility
from arl.visibility.iterators import vis_slice_iter, vis_timeslice_iter, vis_null_iter, \
    vis_wstack_iter
from arl.data.parameters import get_parameter

log = logging.getLogger(__name__)


def imaging_contexts():
    """Contains all the context information for imaging
    
    The fields are:
        predict: Predict function to be used
        invert: Invert function to be used
        image_iterator: Iterator for traversing images
        image_iterator_args: Arguments for image_iter
        vis_iterator: Iterator for traversing visibilities
        vis_iterator_args: Arguments for vis_iter
        inner: The innermost axis
    
    :return:
    """
    contexts = {'2d': {'predict': predict_2d,
                       'invert': invert_2d,
                       'image_iterator': image_null_iter,
                       'image_iterator_args': '',
                       'vis_iterator': vis_null_iter,
                       'vis_iterator_args': '',
                       'inner': 'image'},
                'facets': {'predict': predict_2d,
                           'invert': invert_2d,
                           'image_iterator': image_raster_iter,
                           'image_iterator_args': ['facets'],
                           'vis_iterator': vis_null_iter,
                           'vis_iterator_args': '',
                           'inner': 'image'},
                'facets_slice': {'predict': predict_2d,
                                 'invert': invert_2d,
                                 'image_iterator': image_raster_iter,
                                 'image_iterator_args': ['facets'],
                                 'vis_iterator': vis_slice_iter,
                                 'vis_iterator_args': ['vis_slices'],
                                 'inner': 'vis'},
                'facets_timeslice': {'predict': predict_timeslice_single,
                                     'invert': invert_timeslice_single,
                                     'image_iterator': image_raster_iter,
                                     'image_iterator_args': ['facets'],
                                     'vis_iterator': vis_timeslice_iter,
                                     'vis_iterator_args': ['timeslice'],
                                     'inner': 'image'},
                'facets_wstack': {'predict': predict_wstack_single,
                                  'invert': invert_wstack_single,
                                  'image_iterator': image_raster_iter,
                                  'image_iterator_args': ['facets'],
                                  'vis_iterator': vis_wstack_iter,
                                  'vis_iterator_args': ['wstack'],
                                  'inner': 'vis'},
                'slice': {'predict': predict_2d,
                          'invert': invert_2d,
                          'image_iterator': image_null_iter,
                          'image_iterator_args': '',
                          'vis_iterator': vis_slice_iter,
                          'vis_iterator_args': ['vis_slices'],
                          'inner': 'image'},
                'timeslice': {'predict': predict_timeslice_single,
                              'invert': invert_timeslice_single,
                              'image_iterator': image_null_iter,
                              'image_iterator_args': '',
                              'vis_iterator': vis_timeslice_iter,
                              'vis_iterator_args': ['timeslice'],
                              'inner': 'image'},
                'wstack': {'predict': predict_wstack_single,
                           'invert': invert_wstack_single,
                           'image_iterator': image_null_iter,
                           'image_iterator_args': '',
                           'vis_iterator': vis_wstack_iter,
                           'vis_iterator_args': ['wstack'],
                           'inner': 'image'}}
    
    return contexts


def imaging_context(context='2d'):
    contexts = imaging_contexts()
    assert context in contexts.keys(), context
    return contexts[context]


def make_vis_iter(vis, context, arl_config='arl_config.ini'):
    """Make the visibility iterator for this context

    :param context:
    :param arl_config:
    :return:
    """
    c = imaging_context(context)
    vis_iter = c['vis_iterator']
    keys = c['vis_iterator_args']
    params = {}
    if keys != '':
        for key in keys:
            value = get_parameter(arl_config=arl_config, key=key, section='imaging')
            params[key] = value
        
        return vis_iter(vis, **params)
    else:
        return vis_iter(vis)


def make_image_iter(im, context, arl_config='arl_config.ini'):
    """Make the image iterator for this context

    :param im: Image
    :param context:
    :param arl_config:
    :return:
    """
    c = imaging_context(context)
    image_iter = c['image_iterator']
    keys = c['image_iterator_args']
    params = {}
    if keys != '':
        for key in keys:
            value = get_parameter(arl_config=arl_config, key=key, section='imaging')
            params[key] = value
        
        return image_iter(im, **params)
    else:
        return image_iter(im)


def invert_function(vis, im: Image, dopsf=False, normalize=True, context='2d', inner=None,
                    arl_config='arl_config.ini'):
    """ Invert using algorithm specified by context:

     * 2d: Two-dimensional transform
     * wstack: wstacking with either vis_slices or wstack (spacing between w planes) set
     * wprojection: w projection with wstep (spacing between w places) set, also kernel='wprojection'
     * timeslice: snapshot imaging with either vis_slices or timeslice set. timeslice='auto' does every time
     * facets: Faceted imaging with facets facets on each axis
     * facets_wprojection: facets AND wprojection
     * facets_wstack: facets AND wstacking
     * wprojection_wstack: wprojection and wstacking


    :param vis:
    :param im:
    :param dopsf: Make the psf instead of the dirty image (False)
    :param normalize: Normalize by the sum of weights (True)
    :param context: Imaging context e.g. '2d', 'timeslice', etc.
    :param inner: Inner loop 'vis'|'image'
    :param arl_config:
    :return: Image, sum of weights
    """
    c = imaging_context(context)
    invert = c['invert']
    if inner is None:
        inner = c['inner']
    
    if not isinstance(vis, Visibility):
        svis = coalesce_visibility(vis)
    else:
        svis = vis

    resultimage = create_empty_image_like(im)

    if inner == 'image':
        totalwt = None
        for rows in make_vis_iter(vis, context, arl_config=arl_config):
            if numpy.sum(rows):
                visslice = create_visibility_from_rows(svis, rows)
                sumwt = 0.0
                workimage = create_empty_image_like(im)
                for dpatch in make_image_iter(workimage, context, arl_config=arl_config):
                    result, sumwt = invert(visslice, dpatch, dopsf, normalize=False, arl_config=arl_config)
                    # Ensure that we fill in the elements of dpatch instead of creating a new numpy arrray
                    dpatch.data[...] = result.data[...]
                # Assume that sumwt is the same for all patches
                if totalwt is None:
                    totalwt = sumwt
                else:
                    totalwt += sumwt
                resultimage.data += workimage.data
    else:
        # We assume that the weight is the same for all image iterations
        totalwt = None
        workimage = create_empty_image_like(im)
        for dpatch in make_image_iter(workimage, context, arl_config=arl_config):
            totalwt = None
            for rows in make_vis_iter(svis, context, arl_config=arl_config):
                if numpy.sum(rows):
                    visslice = create_visibility_from_rows(svis, rows)
                    result, sumwt = invert(visslice, dpatch, dopsf, normalize=False, arl_config=arl_config)
                    # Ensure that we fill in the elements of dpatch instead of creating a new numpy arrray
                    dpatch.data[...] += result.data[...]
                    if totalwt is None:
                        totalwt = sumwt
                    else:
                        totalwt += sumwt
            resultimage.data += workimage.data
            workimage.data[...] = 0.0
    
    assert totalwt is not None, "No valid data found for imaging"
    if normalize:
        resultimage = normalize_sumwt(resultimage, totalwt)
    
    return resultimage, totalwt


def predict_function(vis, model: Image, context='2d', inner=None, kernel='2d', arl_config='arl_config.ini') -> \
        Visibility:
    """Predict visibilities using algorithm specified by context
    
     * 2d: Two-dimensional transform
     * wstack: wstacking with either vis_slices or wstack (spacing between w planes) set
     * wprojection: w projection with wstep (spacing between w places) set, also kernel='wprojection'
     * timeslice: snapshot imaging with either vis_slices or timeslice set. timeslice='auto' does every time
     * facets: Faceted imaging with facets facets on each axis
     * facets_wprojection: facets AND wprojection
     * facets_wstack: facets AND wstacking
     * wprojection_wstack: wprojection and wstacking

    
    :param vis:
    :param model: Model image, used to determine image characteristics
    :param context: Imaing context e.g. '2d', 'timeslice', etc.
    :param inner: Inner loop 'vis'|'image'
    :param arl_config:
    :return:


    """
    c = imaging_context(context)
    predict = c['predict']
    if inner is None:
        inner = c['inner']

    if not isinstance(vis, Visibility):
        svis = coalesce_visibility(vis)
    else:
        svis = vis
    
    result = copy_visibility(vis, zero=True)
    
    if inner == 'image':
        for rows in make_vis_iter(svis, context, arl_config=arl_config):
            if numpy.sum(rows):
                visslice = create_visibility_from_rows(svis, rows)
                visslice.data['vis'][...] = 0.0
                # Iterate over images
                for dpatch in make_image_iter(model, context, arl_config=arl_config):
                    result.data['vis'][...] = 0.0
                    result = predict(visslice, dpatch, arl_config=arl_config)
                    svis.data['vis'][rows] += result.data['vis']
    else:
        # Iterate over images
        for dpatch in make_image_iter(model, context, arl_config=arl_config):
            for rows in make_vis_iter(svis, context, arl_config=arl_config):
                if numpy.sum(rows):
                    visslice = create_visibility_from_rows(svis, rows)
                    result.data['vis'][...] = 0.0
                    result = predict(visslice, dpatch, arl_config=arl_config)
                    svis.data['vis'][rows] += result.data['vis']
    
    return svis
