# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from cv_demos.deit.pytorch_deit_classify_16_224_hf import run_deit_classify_224_hf_pytorch

variants = [
    "facebook/deit-base-patch16-224",
    "facebook/deit-base-distilled-patch16-224",
    "facebook/deit-small-patch16-224",
    "facebook/deit-tiny-patch16-224",
]


@pytest.mark.parametrize("variant", variants, ids=variants)
@pytest.mark.deit
def test_deit_classify_224_hf_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_deit_classify_224_hf_pytorch(variant, batch_size=batch_size)
