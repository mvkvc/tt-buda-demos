# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from nlp_demos.dpr.pytorch_dpr_context_encoder import run_dpr_context_encoder_pytorch
from nlp_demos.dpr.pytorch_dpr_question_encoder import run_dpr_question_encoder_pytorch
from nlp_demos.dpr.pytorch_dpr_reader import run_dpr_reader_pytorch

variants_ctx = ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]
variants_qe = ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]
variants_reader = ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]


@pytest.mark.parametrize("variant", variants_ctx, ids=variants_ctx)
@pytest.mark.dpr
def test_dpr_context_encoder_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_dpr_context_encoder_pytorch(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants_qe, ids=variants_qe)
@pytest.mark.dpr
def test_dpr_question_encoder_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_dpr_question_encoder_pytorch(variant, batch_size=batch_size)


@pytest.mark.parametrize("variant", variants_reader, ids=variants_reader)
@pytest.mark.dpr
def test_dpr_reader_pytorch(clear_pybuda, test_device, variant, batch_size):
    run_dpr_reader_pytorch(variant, batch_size=batch_size)
