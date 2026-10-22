import numpy as np
import pytest

from holofocus.learned.misspecification import PrincipalComponents, knn_distance


def test_knn_distance_on_a_small_set():
    reference = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    points = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert knn_distance(points, reference, 1) == pytest.approx([0.0, 3.0])
    assert knn_distance(points, reference, 2) == pytest.approx([1.5, 3.5])
    assert knn_distance(points, reference, 3) == pytest.approx([7.0 / 3.0, 4.0])


def test_knn_distance_chunks_and_errors():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 6)).astype(np.float32)
    points = rng.normal(size=(23, 6))
    full = knn_distance(points, reference, 5)
    assert knn_distance(points, reference, 5, chunk=4) == pytest.approx(full, rel=1e-12)
    brute = np.sort(np.linalg.norm(points[:, None] - reference[None], axis=2), axis=1)[:, :5]
    assert full == pytest.approx(brute.mean(axis=1), rel=1e-9)
    assert len(knn_distance(points[:0], reference, 5)) == 0
    for k in (0, 51):
        with pytest.raises(ValueError):
            knn_distance(points, reference, k)
    with pytest.raises(ValueError):
        knn_distance(points[:, :3], reference, 1)


def test_principal_components_of_an_axis_aligned_gaussian():
    rng = np.random.default_rng(1)
    scale = np.array([2.0, 1.0, 0.1])
    reference = rng.normal(size=(20000, 3)) * scale + np.array([5.0, -1.0, 0.0])
    pca = PrincipalComponents(reference, components=2)
    assert pca.components == 2 and pca.rank == 3
    assert pca.variances == pytest.approx(scale[:2] ** 2, rel=0.03)
    assert pca.explained == pytest.approx(5.0 / 5.01, rel=1e-3)
    point = np.array([[5.0 + 2.0, -1.0, 0.0], [5.0, -1.0 + 3.0, 0.0], [5.0, -1.0, 0.3]])
    assert pca.mahalanobis(point) == pytest.approx([1.0, 3.0, 0.0], abs=0.05)
    assert pca.residual_rms(point) == pytest.approx([0.0, 0.0, 0.3 / np.sqrt(3.0)], abs=0.01)
    assert pca.components_for(0.5) == 1 and pca.components_for(0.99) == 2
    assert pca.components_for(0.999) == 3
    assert PrincipalComponents(reference, variance=0.75).components == 1
    assert PrincipalComponents(reference, variance=0.99).components == 2


def test_principal_components_stop_at_the_rank():
    rng = np.random.default_rng(2)
    flat = rng.normal(size=(500, 2)) @ np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    pca = PrincipalComponents(flat + 7.0, components=4)
    assert pca.rank == 2 and pca.components == 2
    assert np.isfinite(pca.mahalanobis(flat[:5] + 7.0)).all()
    assert PrincipalComponents(flat, variance=1.0).components == 2
    with pytest.raises(ValueError):
        PrincipalComponents(flat)
    with pytest.raises(ValueError):
        PrincipalComponents(flat, components=2, variance=0.9)
    with pytest.raises(ValueError):
        PrincipalComponents(flat, variance=1.5)
    with pytest.raises(ValueError):
        PrincipalComponents(np.ones((10, 3)), components=1)
