from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.processors.utils import contains_nan
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], float32[:], int32, int32, float32[:])",
        "void(float64[:], float64[:], int32, int32, float64[:])",
    ],
    "(n),(m),(),()->(n)",
    **nb_kwargs,
)
def wf_correction(
    w_in: np.ndarray,
    w_corr: np.ndarray,
    start_idx: int,
    stop_idx: int,
    w_out: np.ndarray,
) -> None:
    """Waveform correction.

    Note
    ----
    This processor correct the input waveform by applying a correction.

    Parameters
    ----------
    w_in
        the input waveform.
    w_corr
        correction array.
    start_idx
        index to start the correction.
    stop_idx
        index to stop the correction.
    w_out
        corrected waveform.

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

      "w_corr": {
          "function": "wf_correction",
          "module": "dspeed.processors",
          "args": ["w_in", "w_corr", 0, 2, "w_out"],
          "unit": "ADC"
       }

    """
    w_out[:] = np.nan
    if contains_nan(w_in):
        return

    if contains_nan(w_corr):
        return

    if start_idx < 0:
        msg = "start_idx must be positive"
        raise DSPFatal(msg)
    if start_idx > len(w_in):
        msg = "start_idx must be shorter than input waveform size"
        raise DSPFatal(msg)

    if stop_idx <= 0:
        msg = "stop_idx must be positive"
        raise DSPFatal(msg)
    if stop_idx > len(w_in):
        msg = "stop_idx must be shorter than input waveform size"
        raise DSPFatal(msg)

    if start_idx >= stop_idx:
        msg = "start_idx must be smaller than stop_idx"
        raise DSPFatal(msg)
    if (stop_idx - start_idx) > len(w_corr):
        msg = "stop_idx - start_idx must be smaller than len(w_corr)"
        raise DSPFatal(msg)

    w_out[:] = w_in[:]

    w_out[start_idx:stop_idx] = (
        w_in[start_idx:stop_idx] - w_corr[: stop_idx - start_idx]
    )
