""" Pipelines expressed as dask graphs
"""

from dask import delayed

from arl.data.parameters import get_parameter, set_parameters
from arl.graphs.delayed import create_deconvolve_graph, create_invert_graph, create_residual_graph, \
    create_predict_graph, create_zero_vis_graph_list, create_calibrate_graph_list, \
    create_subtract_vis_graph_list, create_restore_graph


def create_ical_pipeline_graph(vis_graph_list, model_graph: delayed, context='2d',
                               do_selfcal=True, arl_config='arl_config.ini') -> delayed:
    """Create graph for ICAL pipeline

    :param vis_graph_list:
    :param model_graph:
    :param context: imaging context e.g. '2d'
    :param arl_config: Parameters for functions in graphs
    :return:
    """
    psf_graph = create_invert_graph(vis_graph_list, model_graph, dopsf=True, context=context, arl_config=arl_config)
    
    model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
    model_vis_graph_list = create_predict_graph(model_vis_graph_list, model_graph, context=context,
                                                arl_config=arl_config)
    if do_selfcal:
        # Make the predicted visibilities, selfcalibrate against it correcting the gains, then
        # form the residual visibility, then make the residual image
        vis_graph_list = create_calibrate_graph_list(vis_graph_list, model_vis_graph_list, arl_config=arl_config)
        residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
        residual_graph = create_invert_graph(residual_vis_graph_list, model_graph, dopsf=True, context=context,
                                             arl_config=arl_config)
    else:
        # If we are not selfcalibrating it's much easier and we can avoid an unnecessary round of gather/scatter
        # for visibility partitioning such as timeslices and wstack.
        residual_graph = create_residual_graph(vis_graph_list, model_graph, context=context, arl_config=arl_config)
    
    deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph, model_graph, arl_config=arl_config)
    
    nmajor = float(get_parameter(arl_config, "nmajor", 5))
    if nmajor > 1:
        for cycle in range(nmajor):
            if do_selfcal:
                model_vis_graph_list = create_zero_vis_graph_list(vis_graph_list)
                model_vis_graph_list = create_predict_graph(model_vis_graph_list, deconvolve_model_graph,
                                                            context=context, arl_config=arl_config)
                vis_graph_list = create_calibrate_graph_list(vis_graph_list, model_vis_graph_list,
                                                             arl_config=arl_config)
                residual_vis_graph_list = create_subtract_vis_graph_list(vis_graph_list, model_vis_graph_list)
                residual_graph = create_invert_graph(residual_vis_graph_list, model_graph, dopsf=False,
                                                     context=context, arl_config=arl_config)
            else:
                residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph,
                                                       context=context, arl_config=arl_config)
            
            deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph,
                                                             deconvolve_model_graph, arl_config=arl_config)
    residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, context=context,
                                           arl_config=arl_config)
    restore_graph = create_restore_graph(deconvolve_model_graph, psf_graph, residual_graph)
    
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))


def create_continuum_imaging_pipeline_graph(vis_graph_list, model_graph: delayed, context='2d',
                                            arl_config='arl_config.ini') -> delayed:
    """ Create graph for the continuum imaging pipeline.
    
    Same as ICAL but with no selfcal.
    
    :param vis_graph_list:
    :param model_graph:
    :param c_deconvolve_graph: Default: create_deconvolve_graph
    :param c_invert_graph: Default: create_invert_graph
    :param c_residual_graph: Default: Default: create_residual graph
    :param arl_config: Parameters for functions in graphs
    :return:
    """
    psf_graph = create_invert_graph(vis_graph_list, model_graph, dopsf=True, context=context, arl_config=arl_config)
    
    residual_graph = create_residual_graph(vis_graph_list, model_graph, context=context, arl_config=arl_config)
    deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph, model_graph, arl_config=arl_config)
    
    nmajor = int(get_parameter(arl_config, "nmajor", 5))
    if nmajor > 1:
        for cycle in range(nmajor):
            residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, context=context,
                                                   arl_config=arl_config)
            deconvolve_model_graph = create_deconvolve_graph(residual_graph, psf_graph, deconvolve_model_graph,
                                                             arl_config)
    
    residual_graph = create_residual_graph(vis_graph_list, deconvolve_model_graph, context=context,
                                           arl_config=arl_config)
    restore_graph = create_restore_graph(deconvolve_model_graph, psf_graph, residual_graph)
    return delayed((deconvolve_model_graph, residual_graph, restore_graph))


def create_spectral_line_imaging_pipeline_graph(vis_graph_list, model_graph: delayed,
                                                continuum_model_graph=None,
                                                context='2d',
                                                arl_config='arl_config.ini') -> delayed:
    """Create graph for spectral line imaging pipeline

    Uses the ical pipeline after subtraction of a continuum model
    
    :param vis_graph_list: List of visibility graphs
    :param model_graph: Spectral line model graph
    :param continuum_model_graph: Continuum model graph
    :param c_deconvolve_graph: Default: create_deconvolve_graph
    :param c_invert_graph: Default: create_invert_graph,
    :param c_residual_graph: Default: Default: create_residual graph
    :param arl_config: Parameters for functions in graphs
    :return: graphs of (deconvolved model, residual, restored)
    """
    if continuum_model_graph is not None:
        vis_graph_list = create_predict_graph(vis_graph_list, continuum_model_graph, context=context,
                                              arl_config=arl_config)
    d = {}
    d['first_selfcal'] = None
    set_parameters(arl_config, d)
    return create_ical_pipeline_graph(vis_graph_list, model_graph, context=context, arl_config=arl_config)
