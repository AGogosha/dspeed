"""Processors for waveform alignment and interpolation."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.processors.utils import contains_nan
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32, float32, float32, float32[:])",
        "void(float64[:], float64, float64, float64, float64[:])",
    ],
    "(n),(),(),(),(m)",
    **nb_kwargs,
)
def wf_alignment(
    w_in: np.ndarray, centroid: int, shift: int, size: int, w_out: np.ndarray
) -> None:
    """Align waveform.

    Note
    ----
    This processor align the input waveform by setting the centroid position at the center of the output waveform.

    Parameters
    ----------
    w_in
        the input waveform.
    centroid
        centroid.
    shift
        shift.
    size
        size of output waveform.
    w_out
        aligned waveform.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_align:
          function: wf_alignment
          module: dspeed.processors
          args:
            - waveform
            - centroid
            - shift
            - size
            - wf_align

    """
    w_out[:] = np.nan

    if contains_nan(w_in):
        return

    if np.isnan(centroid):
        msg = "centroid is nan"
        raise DSPFatal(msg)

    if np.isnan(shift):
        msg = "shift is nan"
        raise DSPFatal(msg)
    if shift < 0:
        msg = "shift must be positive"
        raise DSPFatal(msg)
    if shift > len(w_in):
        msg = "shift must be shorter than input waveform size"
        raise DSPFatal(msg)

    if np.isnan(size):
        msg = "size is nan"
        raise DSPFatal(msg)
    if size <= 0:
        msg = "size must be positive"
        raise DSPFatal(msg)
    if size > len(w_in):
        msg = "size must be shorter than input waveform size"
        raise DSPFatal(msg)

    if (centroid >= size / 2) and (centroid < len(w_in) - size / 2):
        w_out[:] = w_in[int(centroid - size / 2) : int(centroid + size / 2)]
    elif (centroid > size / 2 - shift) and (centroid < size / 2):
        ss = int((size + 1) / 2 - centroid)
        w_out[:ss] = w_in[0]
        w_out[ss:] = w_in[: int(centroid + size / 2)]
    else:
        w_out[:] = w_in[:size]
