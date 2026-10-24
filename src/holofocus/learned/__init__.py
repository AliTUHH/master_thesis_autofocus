"""Stage 4: learned autofocus, a CNN estimating ``dz01`` with a Gaussian uncertainty.

Stage 7: its posterior by neural posterior estimation, :mod:`holofocus.learned.npe` (needs
sbi, the optional extra ``npe``, and is therefore not imported here) and the sbi-free
:mod:`holofocus.learned.posterior`.
"""

from .compare import compare_methods, offset_estimate
from .config import (
    DensityConfig,
    EmbeddingConfig,
    LearnedConfig,
    ModelConfig,
    NPEConfig,
    NPETrainingConfig,
    RepresentationConfig,
    TrainingConfig,
)
from .data import d4_transform, load_inputs, random_d4
from .inference import LearnedAutofocus, predict_split, read_predictions, summarize_predictions
from .metrics import bootstrap_ci, calibration_scale, interval_metrics, point_metrics
from .models import ImageCNN, PolarSpectrumCNN, SpectrumCNN, build_model, gaussian_nll
from .posterior import GridPosterior, calibration_summary, summarize_npe_predictions
from .representations import Representation
from .training import read_history, train_model

__all__ = [
    "DensityConfig",
    "EmbeddingConfig",
    "GridPosterior",
    "ImageCNN",
    "LearnedAutofocus",
    "LearnedConfig",
    "ModelConfig",
    "NPEConfig",
    "NPETrainingConfig",
    "PolarSpectrumCNN",
    "Representation",
    "RepresentationConfig",
    "SpectrumCNN",
    "TrainingConfig",
    "bootstrap_ci",
    "build_model",
    "calibration_scale",
    "calibration_summary",
    "compare_methods",
    "d4_transform",
    "gaussian_nll",
    "interval_metrics",
    "load_inputs",
    "offset_estimate",
    "point_metrics",
    "predict_split",
    "random_d4",
    "read_history",
    "read_predictions",
    "summarize_npe_predictions",
    "summarize_predictions",
    "train_model",
]
