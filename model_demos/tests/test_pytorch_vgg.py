# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.vgg.pytorch_vgg_hf import run_vgg_19_hf_pytorch
from cv_demos.vgg.pytorch_vgg_osmr import run_vgg_osmr_pytorch
from cv_demos.vgg.pytorch_vgg_timm import run_vgg_bn19_timm_pytorch
from cv_demos.vgg.pytorch_vgg_torchhub import run_vgg_bn19_torchhub_pytorch

variants1 = ["vgg11", "vgg13", "vgg16", "vgg19", "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]
variants2 = ["vgg11", "vgg13", "vgg16", "vgg19", "bn_vgg19", "bn_vgg19b"]


@pytest.mark.parametrize("variant", variants1, ids=variants1)
@pytest.mark.vgg
def test_vgg_19_hf_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_vgg_19_hf_pytorch(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants1, ids=variants1)
@pytest.mark.vgg
def test_vgg_bn19_timm_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_vgg_bn19_timm_pytorch(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants1, ids=variants1)
@pytest.mark.vgg
def test_vgg_bn19_torchhub_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_vgg_bn19_torchhub_pytorch(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants2, ids=variants2)
@pytest.mark.vgg
def test_vgg_osmr_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_vgg_osmr_pytorch(variant, batch_size=batch_size)
