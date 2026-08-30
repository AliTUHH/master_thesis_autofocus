# -*- coding: utf-8 -*-
import numpy as np

def calculate_mae(predictions, true_values):
    """
    Calculate the mean absolute error between predictions and reference values.

    Both arguments must be NumPy-compatible arrays with matching shapes. The
    returned scalar uses the same physical unit as the supplied values.
    """
    return np.mean(np.abs(predictions - true_values))
