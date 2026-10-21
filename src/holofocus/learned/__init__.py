"""Stage 4: learned autofocus, a CNN estimating ``dz01`` with a Gaussian uncertainty."""

from .compare import compare_methods, offset_estimate
from .config import LearnedConfig, ModelConfig, RepresentationConfig, TrainingConfig
from .data import d4_transform, load_inputs, random_d4
from .inference import LearnedAutofocus, predict_split, read_predictions, summarize_predictions
from .metrics import bootstrap_ci, calibration_scale, interval_metrics, point_metrics
from .models import ImageCNN, SpectrumCNN, build_model, gaussian_nll
from .representations import Representation
from .training import read_history, train_model

__all__ = [
    "ImageCNN",
    "LearnedAutofocus",
    "LearnedConfig",
    "ModelConfig",
    "Representation",
    "RepresentationConfig",
    "SpectrumCNN",
    "TrainingConfig",
    "bootstrap_ci",
    "build_model",
    "calibration_scale",
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
    "summarize_predictions",
    "train_model",
]
