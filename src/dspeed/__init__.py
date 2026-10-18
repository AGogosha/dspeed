r"""The *dspeed* signal processing framework is responsible for running a variety
of discrete signal processors on data.
"""

from ._version import version as __version__
from .build_dsp import build_dsp
from .logging import log
from .processing_chain import ProcessingChain, build_processing_chain

__all__ = [
    "ProcessingChain",
    "__version__",
    "build_dsp",
    "build_processing_chain",
    "log",
]
