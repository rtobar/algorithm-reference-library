""" Unit tests for pipelines


"""

import logging
import unittest

from arl.data.parameters import get_parameter, get_parameters, set_parameters, clear_parameters

log = logging.getLogger(__name__)


class TestParameters(unittest.TestCase):
    
    def test_setgetparameters(self):
        config_file = 'test_parameter.ini'
        
        set_parameters(config_file, {'npixel': 256, 'cellsize': 0.1, 'spectral_mode': 'mfs'}, 'mine')
        set_parameters(config_file, {'gain': 0.1, 'algorithm': 'mmclean'}, 'yours')
        assert len(get_parameters(config_file, 'mine')) == 3
        assert len(get_parameters(config_file, 'yours')) == 2

        assert float(get_parameter(config_file, 'cellsize', 0.0, section='mine')) == 0.1
        assert get_parameter(config_file, 'spectral_mode', 'channels', section='mine') == 'mfs'
        assert get_parameter(config_file, 'null_mode', 'mfs', section='mine') == 'mfs'
        assert get_parameter(config_file, 'foo', 'bar', section='mine') == 'bar'
        assert get_parameter(config_file, 'foo', section='mine') is None
        assert get_parameter(None, 'foo', 'bar', section='mine') == 'bar'

if __name__ == '__main__':
    unittest.main()
