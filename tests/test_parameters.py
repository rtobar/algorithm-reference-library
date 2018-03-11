""" Unit tests for pipelines


"""

import unittest

from arl.data.parameters import get_parameter, set_parameters

import logging

log = logging.getLogger(__name__)


class TestParameters(unittest.TestCase):
    def setUp(self):
        self.parameters = {'npixel': 256, 'cellsize': 0.1, 'spectral_mode': 'mfs'}
        self.config_file = 'test_config.ini'
    
    def test_setgetparameters(self):
        set_parameters(self.config_file, self.parameters)
    
        def t1(arl_config):
            assert float(get_parameter(arl_config, 'cellsize')) == 0.1
            assert get_parameter(arl_config, 'spectral_mode', 'channels') == 'mfs'
            assert get_parameter(arl_config, 'null_mode', 'mfs') == 'mfs'
            assert get_parameter(arl_config, 'foo', 'bar') == 'bar'
            assert get_parameter(arl_config, 'foo') is None
            assert get_parameter(None, 'foo', 'bar') == 'bar'
    
        t1(self.config_file)
    

if __name__ == '__main__':
    unittest.main()
