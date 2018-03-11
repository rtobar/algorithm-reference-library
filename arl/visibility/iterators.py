""" Visibility iterators for iterating through a BlockVisibility or Visibility.

A typical use would be to make a sequence of snapshot images::

    for rows in vis_timeslice_iter(vt):
        visslice = create_visibility_from_rows(vt, rows)
        dirtySnapshot = create_image_from_visibility(visslice, npixel=512, cellsize=0.001, npol=1)
        dirtySnapshot, sumwt = invert_2d(visslice, dirtySnapshot)


"""

import logging
from typing import Union

import numpy

from arl.data.data_models import Visibility, BlockVisibility

log = logging.getLogger(__name__)

def vis_null_iter(vis: Visibility) -> numpy.ndarray:
    """One time iterator returning true for all rows
    
    :param vis:
    :return:
    """
    yield numpy.ones_like(vis.time, dtype=bool)


def vis_timeslice_iter(vis: Visibility, timeslice='auto', vis_slices=None) -> numpy.ndarray:
    """ W slice iterator

    :param wstack: wstack (wavelengths)
    :param vis_slices: Number of slices (second in precedence to wstack)
    :return: Boolean array with selected rows=True
    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    timemin = numpy.min(vis.time)
    timemax = numpy.max(vis.time)
    
    if timeslice == 'auto':
        boxes = numpy.unique(vis.time)
        timeslice = 0.1
    elif timeslice is None:
        timeslice = timemax - timemin
        boxes = [0.5*(timemax+timemin)]
    elif isinstance(timeslice, float) or isinstance(timeslice, int):
        boxes = numpy.arange(timemin, timemax, timeslice)
    else:
        assert vis_slices is not None, "Time slicing not specified: set either timeslice or vis_slices"
        boxes = numpy.linspace(timemin, timemax, vis_slices)
        if vis_slices > 1:
            timeslice = boxes[1] - boxes[0]
        else:
            timeslice = timemax - timemin

    for box in boxes:
        rows = numpy.abs(vis.time - box) <= 0.5 * timeslice
        yield rows


def vis_wstack_iter(vis: Visibility, wstack=None, vis_slices=None) -> numpy.ndarray:
    """ W slice iterator

    :param wstack: wstack (wavelengths)
    :param vis_slices: Number of slices (second in precedence to wstack)
    :return: Boolean array with selected rows=True
    """
    assert isinstance(vis, Visibility), vis
    wmaxabs = numpy.max(numpy.abs(vis.w))
    
    if wstack is None:
        assert vis_slices is not None, "w slicing not specified: set either wstack or vis_slices"
        boxes = numpy.linspace(-wmaxabs, wmaxabs, vis_slices)
        if vis_slices > 1:
            wstack = boxes[1] - boxes[0]
        else:
            wstack = 2 * wmaxabs
    else:
        vis_slices = 1 + 2 * numpy.round(wmaxabs / float(wstack)).astype('int')
        boxes = numpy.linspace(- wmaxabs, +wmaxabs, vis_slices)
        if vis_slices > 1:
            wstack = boxes[1] - boxes[0]
        else:
            wstack = 2 * wmaxabs
    
    for box in boxes:
        rows = numpy.abs(vis.w - box) < 0.5 * wstack
        yield rows


def vis_slice_iter(vis: Union[Visibility, BlockVisibility], step=None, vis_slices=None) -> numpy.ndarray:
    """ Iterates in slices

    :param step: Size of step to be iterated over (in rows)
    :param vis_slices: Number of slices (second in precedence to step)
    :return: Boolean array with selected rows=True

    """
    assert isinstance(vis, Visibility) or isinstance(vis, BlockVisibility), vis
    
    vis_slices=int(vis_slices)
    
    if step is None:
        assert vis_slices is not None, "vis slicing not specified: set either step or vis_slices"
        step = vis.nvis // vis_slices
        
    assert step > 0
    for row in range(0, vis.nvis, step):
        rows = vis.nvis * [False]
        for r in range(row, min(row+step, vis.nvis)):
            rows[r] = True
        yield rows
