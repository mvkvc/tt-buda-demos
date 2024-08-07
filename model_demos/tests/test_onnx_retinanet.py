# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.retinanet.onnx_retinanet_r101 import run_retinanet_r101_640x480_onnx


@pytest.mark.retinanet
def test_retinanet_onnx(clear_pybuda, test_device, batch_size):
    run_retinanet_r101_640x480_onnx(batch_size=batch_size)
