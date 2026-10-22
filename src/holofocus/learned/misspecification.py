"""Distances of observations to a training set, for checks of misspecification (Stage 8).

[6] (Sec. 3.4) recommends checking before inference whether an observation lies where the
simulations lie, e.g. as an outlier in the space of the embedding. Two kinds of distance of a
point to the training points, in any space (the network input, an embedding):

* :func:`knn_distance`: the mean Euclidean distance to the ``k`` nearest training points; no
  model of the distribution, it follows the local density of the training points;
* :class:`PrincipalComponents`: the Mahalanobis distance in the leading principal components
  of the training points and the RMS of the residual outside them; a Gaussian model of the
  bulk, blind to how the points are spread inside it.

Neither has a scale of its own. Distances of held-out samples of the same simulator (the test
split) to the same training set give it: the percentile of an observation among them
(:func:`holofocus.learned.sim2real.percentile_rank`) says how many simulated holograms the
networks never saw lie farther away.
"""

from __future__ import annotations

import numpy as np

RANK_TOLERANCE = 1e-10
"""Principal components with a variance below this fraction of the largest count as zero; an
embedding can use fewer directions than it has features."""


def knn_distance(points: np.ndarray, reference: np.ndarray, k: int, chunk: int = 256) -> np.ndarray:
    """Mean Euclidean distance of every row of ``points`` ``(N, D)`` to its ``k`` nearest rows
    of ``reference`` ``(M, D)``, in float64; ``(N,)``."""
    reference = np.asarray(reference, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or reference.ndim != 2 or points.shape[1] != reference.shape[1]:
        raise ValueError("points and reference must have shapes (N, D) and (M, D)")
    if not 1 <= k <= len(reference):
        raise ValueError(f"k = {k} must lie between 1 and the {len(reference)} reference points")
    norms = (reference**2).sum(axis=1)
    out = []
    for start in range(0, len(points), chunk):
        x = points[start : start + chunk]
        squared = (x**2).sum(axis=1)[:, None] + norms[None, :] - 2.0 * x @ reference.T
        distance = np.sqrt(np.maximum(squared, 0.0))
        out.append(np.partition(distance, k - 1, axis=1)[:, :k].mean(axis=1))
    return np.concatenate(out) if out else np.zeros(0)


class PrincipalComponents:
    """Principal components of the training points ``(M, D)``: the first ``components``, or as
    many as hold the fraction ``variance`` of the total variance, at most the numerical rank."""

    def __init__(
        self, reference: np.ndarray, components: int | None = None, variance: float | None = None
    ) -> None:
        if (components is None) == (variance is None):
            raise ValueError("give either components or variance")
        x = np.asarray(reference, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2:
            raise ValueError("reference must have shape (M, D) with M >= 2")
        self.mean = x.mean(axis=0)
        centred = x - self.mean
        values, vectors = np.linalg.eigh(centred.T @ centred / (len(x) - 1))
        values, vectors = np.maximum(values[::-1], 0.0), vectors[:, ::-1]
        self.dimension = x.shape[1]
        self.total_variance = float(values.sum())
        if self.total_variance <= 0:
            raise ValueError("the reference points do not vary")
        self.rank = int((values > RANK_TOLERANCE * values[0]).sum())
        cumulative = np.cumsum(values) / self.total_variance
        self.cumulative = cumulative
        if components is None:
            if not 0 < variance <= 1:
                raise ValueError("variance must lie in (0, 1]")
            components = int(np.searchsorted(cumulative, variance - 1e-12) + 1)
        if components < 1:
            raise ValueError("components must be positive")
        self.components = min(int(components), self.rank)
        self.variances = values[: self.components]
        self.vectors = vectors[:, : self.components]
        self.explained = float(cumulative[self.components - 1])

    def components_for(self, variance: float) -> int:
        """How many components hold the fraction ``variance`` of the total variance."""
        return int(np.searchsorted(self.cumulative, variance - 1e-12) + 1)

    def scores(self, points: np.ndarray) -> np.ndarray:
        """Coordinates in the units of the standard deviation of each component ``(N, m)``."""
        centred = np.asarray(points, dtype=np.float64) - self.mean
        return centred @ self.vectors / np.sqrt(self.variances)

    def mahalanobis(self, points: np.ndarray) -> np.ndarray:
        return np.sqrt((self.scores(points) ** 2).sum(axis=1))

    def residual_rms(self, points: np.ndarray) -> np.ndarray:
        """RMS over the ``D`` coordinates of what the components do not describe."""
        centred = np.asarray(points, dtype=np.float64) - self.mean
        residual = centred - (centred @ self.vectors) @ self.vectors.T
        return np.sqrt((residual**2).mean(axis=1))
