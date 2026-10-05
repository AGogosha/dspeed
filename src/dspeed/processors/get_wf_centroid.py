"""Processors estimating waveform centroids."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.processors.utils import contains_nan
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32[:])",
        "void(float64[:], float64, float64[:])",
    ],
    "(n),()->()",
    **nb_kwargs,
)
def get_wf_centroid(w_in: np.ndarray, shift: int, centroid: int) -> None:
    """Calculate waveform centroid.

    Note
    ----
    This processor calculate the centroid position when provided the convolution product with a step function.

    Parameters
    ----------
    w_in
        the input waveform.
    shift
        shift.
    centroid
        centroid position.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        centroid:
          function: get_wf_centroid
          module: dspeed.processors
          args:
            - waveform
            - shift
            - centroid

    """
    centroid[0] = np.nan

    if contains_nan(w_in):
        return

    if np.isnan(shift):
        msg = "shift is nan"
        raise DSPFatal(msg)
    if shift < 0:
        msg = "shift must be positive"
        raise DSPFatal(msg)
    if shift > len(w_in) - 1:
        msg = "shift must be shorter than input waveform size"
        raise DSPFatal(msg)

    i_min = np.argmin(w_in)
    i_max = np.argmax(w_in)
    if i_min == i_max:
        return

    c_a = -1
    c_b = -1

    # search from minimum to maximum for last negative and first positive
    for i in range(i_min, i_max + np.sign(i_max - i_min), np.sign(i_max - i_min)):
        if w_in[i] < 0:
            c_a = i
        if w_in[i] > 0 and c_b == -1:
            c_b = i

    if c_a == -1 or c_b == -1:
        return

    centroid[0] = round((c_a + c_b) / 2) + shift
