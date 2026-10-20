"""CPU smoke test of the paper code: holowizard ASRM on the measured spider-hair hologram.

Runs the first stages of the ASRM schedule used by ``examples/asrm/spider_hair.py`` of
holowizard 3.0.6 (700 iterations at 16x down-sampling, then 300 at 4x) and saves the
hologram next to the reconstructed phase. The hologram ships with the Zenodo record of [1]
(10.5281/zenodo.8349365, ``Examples/data/holograms/spider_hair.tiff``, MIT licence).

Needs a separate environment (holowizard pins Python 3.11), e.g.::

    uv venv --python 3.11 /tmp/hw/venv311
    VIRTUAL_ENV=/tmp/hw/venv311 uv pip install holowizard==3.0.6
    /tmp/hw/venv311/bin/python scripts/holowizard_asrm_cpu.py --data spider_hair.tiff
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import holowizard.core  # noqa: E402
import holowizard.core.utils.fileio as fileio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from holowizard.core.api.functions.single_projection.reconstruction import (  # noqa: E402
    reconstruct,
)
from holowizard.core.api.parameters import (  # noqa: E402
    BeamSetup,
    DataDimensions,
    Measurement,
    Options,
    Padding,
    RecoParams,
    Regularization,
)
from holowizard.core.logging.logger import Logger  # noqa: E402


def options_for(down_sampling: int, first: bool, padding: Padding) -> Options:
    """Stage options of holowizard's spider-hair example (16x warm-up, then 4x)."""
    stage = Options(
        regularization_object=Regularization(
            iterations=700 if first else 300,
            update_rate=0.9 if first else 1.1,
            l2_weight=0.0 + 10.0j,
            gaussian_filter_fwhm=2.0 + 0.0j if first else 2.0 + 8.0j,
        ),
        nesterov_object=Regularization(
            update_rate=1.0, gaussian_filter_fwhm=(8.0 + 8.0j) if first else (16.0 + 16.0j)
        ),
        verbose_interval=100,
        padding=deepcopy(padding),
    )
    stage.padding.down_sampling_factor = down_sampling
    return stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="spider_hair.tiff")
    parser.add_argument("--stages", type=int, nargs="+", default=[16, 4])
    parser.add_argument("--z01-mm", type=float, default=79.45, help="value of the example")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("stage1-holowizard-asrm-cpu.png"))
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/holowizard-logs"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    Logger.current_log_level = Logger.level_num_loss
    Logger.configure(session_name="asrm_cpu", working_dir=str(args.log_dir) + "/")

    data = fileio.load_img_data(str(args.data))
    setup = BeamSetup(energy=11.0, px_size=0.0065, z02=19_661.0)
    measurements = [Measurement(data_path=str(args.data), data=data, z01=args.z01_mm)]
    padding = Padding(
        padding_mode=Padding.PaddingMode.MIRROR_ALL,
        padding_factor=7.0,
        down_sampling_factor=args.stages[0],
        cutting_band=0,
        a0=0.98,
    )
    options = [options_for(s, i == 0, padding) for i, s in enumerate(args.stages)]
    dims = DataDimensions(total_size=data.shape, fov_size=data.shape, window_type="blackman")
    params = RecoParams(
        beam_setup=setup,
        output_path="",
        measurements=measurements,
        reco_options=options,
        data_dimensions=dims,
    )

    start = time.perf_counter()
    result, loss = reconstruct(params, viewer=[])
    seconds = time.perf_counter() - start
    # holowizard's process_image applies torch.rot90 to the input, so the result is rotated.
    phase = np.rot90(result.real.cpu().numpy(), -1)
    print(
        f"device {holowizard.core.torch_running_device_name}, {args.threads} threads, "
        f"stages {args.stages}: {seconds:.0f} s, result {phase.shape}, "
        f"phase [{phase.min():.2f}, {phase.max():.2f}] rad, final loss {float(loss[-1]):.3e}"
    )

    extent = (0, data.shape[1], data.shape[0], 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), layout="constrained")
    im = axes[0].imshow(data, cmap="gray", extent=extent)
    axes[0].set_title(f"Measured hologram (flat-field corrected), {data.shape[0]}² px")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    im = axes[1].imshow(phase, cmap="gray", extent=extent)
    axes[1].set_title(
        f"holowizard ASRM phase / rad, stages {'x, '.join(map(str, args.stages))}x\n"
        f"CPU, {args.threads} threads: {seconds:.0f} s, {phase.shape[0]}² px"
    )
    fig.colorbar(im, ax=axes[1], shrink=0.85)
    for ax in axes:
        ax.set_xlabel("detector pixel")
    fig.suptitle(f"Spider hair, 11 keV, z01 = {args.z01_mm} mm, z02 = 19.661 m ([1] Table 3)")
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
