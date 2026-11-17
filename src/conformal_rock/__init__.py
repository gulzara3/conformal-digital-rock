"""conformal_rock: conformal uncertainty and risk-controlled triage for digital-rock property prediction.

Companion code for Alam & Khan, "Conformal uncertainty quantification and risk-controlled simulation
triage for deep-learning prediction of pore-scale properties from micro-CT images".

Submodules
----------
config     Experiment settings used for every result in the paper.
conformal  Split, normalized, CQR and Mondrian conformal prediction; exact coverage law; joint box.
shift      Conformal out-of-distribution guard and weighted conformal prediction.
triage     Learn-then-Test and conformal-risk-control thresholds for accept-or-simulate decisions.
io         Loading released predictions, splits and the checksum manifest.
metrics    Point-accuracy and coverage metrics.
data       Rebuilding the working dataset from the public DeePore archive (needs h5py and torch).
model      The 3D residual network (needs torch).
"""

__version__ = "2.0.0"

from .config import CFG, PROP_NAMES, PROPS  # noqa: F401
