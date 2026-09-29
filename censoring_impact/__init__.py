"""
censoring_impact: how much do reported discrimination and calibration of fixed prediction models depend on
treating patients who leave follow-up as non-informatively censored?

Implements the fixed-model censoring sensitivity analysis of Fang et al., "How much is the non-informative
censoring assumption worth? A fixed-model sensitivity analysis of time-updated mortality prediction in
hemodialysis".
"""
__version__ = '0.1.0'

from .analysis import Config, Results, run          # noqa: F401
from .core import Arm                               # noqa: F401
