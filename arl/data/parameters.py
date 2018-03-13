"""We use the standard configparser mechanism for arguments. For example::

    kernelname = get_parameter(arl_config, "kernel", "2d")
    oversampling = float(get_parameter(arl_config, "oversampling", 1))
    padding = int(get_parameter(arl_config, "padding", 2))

All functions possess an API which is always of the form::

      def processing_function(idatastruct1, idatastruct2, ..., arl_config='arl_config.ini'):
         return odatastruct1, odatastruct2,... other

Inside a function, the values are retrieved can be accessed directly from the
arl_config.ini, or if a default is needed a function can be used::

    log = get_parameter(arl_config, 'log', None)

Function parameters should obey a consistent naming convention:

A typical file would look like::

    [imaging]
    npixel = 576
    nchan = 1
    reffrequency = 100000000.0
    facets = 9
    padding = 8
    oversampling = 1
    kernel = 2d
    wstep = 4.0
    wstack = 4.0
    vis_slices = 10
    timeslice = auto
    timeslice = auto

The typical naming for argument is:

=======  =======
Name     Meaning
=======  =======
vis      Name of Visibility
sc       Name of Skycomponent
gt       Name of GainTable
conf     Name of Configuration
im       Name of input image
qa       Name of quality assessment
log      Name of processing log
=======  =======

If a function argument has a better, more descriptive name e.g. normalised_gt, newphasecentre, use it.

Keyword=value pairs should have descriptive names. The names should be lower case with underscores to separate words:

====================    ==================================  ========================================================
Name                    Meaning                             Example
====================    ==================================  ========================================================
loop_gain               Clean loop gain                     0.1
niter                   Number of iterations                10000
eps                     Fractional tolerance                1e-6
threshold               Absolute threshold                  0.001
fractional_threshold    Threshold as fraction of e.g. peak  0.1
G_solution_interval     Solution interval for G term        100
phaseonly               Do phase-only solutions             True
phasecentre             Phase centre (usually as SkyCoord)  SkyCoord("-1.0d", "37.0d", frame='icrs', equinox='J2000')
spectral_mode           Visibility processing mode          'mfs' or 'channel'
====================    ==================================  ========================================================

"""

import logging
import os
import ast
import configparser

log = logging.getLogger(__name__)


def arl_path(path):
    """Converts a path that might be relative to ARL root into an
    absolute path::

        arl_path('data/models/SKA1_LOW_beam.fits')
        '/Users/timcornwell/Code/algorithm-reference-library/data/models/SKA1_LOW_beam.fits'

    :param path:
    :return: absolute path
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arlhome = os.getenv('ARL', project_root)
    return os.path.join(arlhome, path)


def get_parameter(arl_config, key=None, default=None, section='DEFAULT'):
    """ Get a specified named value from the given section

    :param config: Parameter dictionary
    :param key: Key e.g. 'loop_gain'
    :param default: Default value
    :param section: Section name e.g. 'imaging' or 'calibration'
    :return: result
    """

    if arl_config is None:
        return default

    config = configparser.ConfigParser()
    config.read(arl_config)
    if section in config:
        if key in config[section]:
            try:
                result = ast.literal_eval(config[section].get(key, default))
            except SyntaxError:
                result = config[section].get(key, default)
            except ValueError:
                result = config[section].get(key, default)
            return result
        else:
            return default
    else:
        return default
    
def get_parameters(arl_config, section='DEFAULT'):
    """ Get all values for a section

    :param config: Parameter dictionary
    :param section: Section name e.g. 'imaging' or 'calibration'
    :return: result
    """

    config = configparser.ConfigParser()
    config.read(arl_config)
    result = {}
    for key in config[section].keys():
        try:
            result[key] = ast.literal_eval(config[section][key])
        except SyntaxError:
            result[key] = config[section][key]
        except ValueError:
            result[key] = config[section][key]

    return result

def clear_parameters(arl_config, section='DEFAULT'):
    """Clear a section of the parameters

    :param arl_config: A config file
    :param section: Section to write into
    """
    config = configparser.ConfigParser()
    config.remove_section(section)
    with open(arl_config, 'w') as configfile:
        config.write(configfile)

def set_parameters(arl_config, dict, section='DEFAULT'):
    """ Set a parameter

    Write the dictionary dict into a configuration file

    :param arl_config: A config file
    :param dict:, dictionary of key:values
    :param section: Section to write into
    :param reset: clear before retrieving
    :return: result
    """
    config = configparser.ConfigParser()
    config.read(arl_config)

    config[section] = dict
    with open(arl_config, 'w') as configfile:
        config.write(configfile)