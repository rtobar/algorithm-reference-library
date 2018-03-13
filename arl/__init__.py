""" Algorithm Reference Library

"""
from arl.calibration import calibration_default_args
from arl.graphs import graphs_default_args
from arl.image import image_default_args
from arl.imaging import imaging_default_args
from arl.pipelines import pipelines_default_args
from arl.visibility import visibility_default_args

def arl_default_args(arl_config='arl_config.ini'):

    calibration_default_args(arl_config)
    graphs_default_args(arl_config)
    image_default_args(arl_config)
    imaging_default_args(arl_config)
    pipelines_default_args(arl_config)
    visibility_default_args(arl_config)