import numpy as np
import pytest
import torch

from holofocus.asrm import stage_grid
from holofocus.measured import MEASURED_SAMPLES, bin_mean, load_measured

# [1] Table 3: Fresnel numbers of the example setups on the full 2048^2 detector.
FRESNEL_TABLE3 = {
    "spider_hair": 7.790e-5,
    "tooth": 1.231e-4,
    "cactus_needle": 4.347e-4,
    "magnesium_wire": 4.678e-4,
}


def test_reference_values_follow_paper2_table1():
    assert list(MEASURED_SAMPLES) == ["spider_hair", "tooth", "cactus_needle", "magnesium_wire"]
    for sample in MEASURED_SAMPLES.values():
        assert sample.confidence == pytest.approx(5e-3)
        lower, upper = sample.simplex
        assert lower < sample.z01_nm < upper and lower < sample.z01_visual < upper
        guess = sample.geometry()
        assert guess.z01 == pytest.approx(0.5 * (lower + upper))
        assert sample.geometry(4).fresnel_number == pytest.approx(16 * guess.fresnel_number)


@pytest.mark.parametrize("name", list(FRESNEL_TABLE3))
def test_fresnel_numbers_match_paper1_table3(name):
    sample = MEASURED_SAMPLES[name]
    fr = sample.geometry(z01=sample.z01_example).fresnel_number
    assert fr == pytest.approx(FRESNEL_TABLE3[name], rel=2e-3)


def test_cutting_band_keeps_the_array_size_of_the_detector():
    tooth = MEASURED_SAMPLES["tooth"]
    n = (2048 - 2 * tooth.cutting_band) // 4
    assert stage_grid((n, n), 1, tooth.padding_for(2.0)).shape == (1024, 1024)
    assert stage_grid((n, n), 4, tooth.padding_for()).shape == (512, 512)
    assert MEASURED_SAMPLES["spider_hair"].padding_for() == 7.0


def test_bin_mean_and_loading(tmp_path):
    from PIL import Image

    a = np.arange(16.0).reshape(4, 4)
    np.testing.assert_array_equal(bin_mean(a, 2), [[2.5, 4.5], [10.5, 12.5]])
    assert bin_mean(a, 1) is a
    with pytest.raises(ValueError):
        bin_mean(a, 3)

    g = np.random.default_rng(0)
    image = (1 + 0.1 * g.standard_normal((2048, 2048))).astype(np.float32)
    for name in ("spider_hair", "tooth"):
        Image.fromarray(image, mode="F").save(tmp_path / f"{name}.tiff")
    spider = load_measured(tmp_path, "spider_hair", binning=4)
    assert spider.shape == (512, 512) and spider.dtype == torch.float32
    np.testing.assert_allclose(spider.numpy(), bin_mean(image.astype(np.float64), 4), rtol=1e-6)
    tooth = load_measured(tmp_path, MEASURED_SAMPLES["tooth"], binning=4)
    assert tooth.shape == (472, 472)
    np.testing.assert_allclose(tooth[0, 0].item(), image[80:84, 80:84].mean(), rtol=1e-6)
