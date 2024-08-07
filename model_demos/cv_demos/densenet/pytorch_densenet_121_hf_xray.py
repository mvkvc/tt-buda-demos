# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


# DenseNet-121 X-Ray Demo

import os

import numpy as np
import pybuda
import requests
import torch
import torchvision
import torchvision.transforms
import torchxrayvision as xrv
from PIL import Image
from pybuda._C.backend_api import BackendDevice

torch.multiprocessing.set_sharing_strategy("file_system")


def get_input_img():

    # Get image
    url = "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
    sample_image = Image.open(requests.get(url, stream=True).raw)

    # Preprocess
    img = xrv.datasets.normalize(np.array(sample_image), 255)

    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")

    # Add color channel
    img = img[None, :, :]
    transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    img = transform(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    return img_tensor


def run_densenet_121_hf_xray_pytorch(batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    # Fallbacks to CPU output normalization of the model. Contains problematic indexing which cause dynamic shapes in model, thus, doing this on host.
    compiler_cfg.enable_tm_cpu_fallback = True
    # Does constant prop on TVM side
    compiler_cfg.enable_tvm_constant_prop = True
    # Fallbacks adv_index to CPU. Used to normalize outputs using threshold extracted as model param (part of output normalization).
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    os.environ["PYBUDA_DISABLE_CONSTANT_FOLDING"] = "1"
    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.balancer_policy = "Ribbon"
            compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
        else:
            compiler_cfg.balancer_policy = "CNN"
            os.environ["PYBUDA_PAD_SPARSE_MM"] = "{25:32}"

    # Load input image
    img_tensor = get_input_img()
    n_img_tensor = [img_tensor] * batch_size
    batch_tensor = torch.cat(n_img_tensor, dim=0)

    # Create PyBuda module from PyTorch model
    model_name = "densenet121-res224-all"
    model = xrv.models.get_model(model_name)  # , from_hf_hub=True
    tt_model = pybuda.PyTorchModule("densnet121_pt", model)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_tensor]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    preds = {k: v for k, v in zip(xrv.datasets.default_pathologies, output[0].value().detach().numpy())}

    # Print output
    print(preds)


if __name__ == "__main__":
    run_densenet_121_hf_xray_pytorch()
