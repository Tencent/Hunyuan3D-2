# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
from diffusers import StableDiffusionUpscalePipeline

class Image_Super_Net():
    def __init__(self, config):
        self.up_pipeline_x4 = StableDiffusionUpscalePipeline.from_pretrained(
                        'stabilityai/stable-diffusion-x4-upscaler',
                        torch_dtype=torch.float16,
                    ).to(config.device)
        self.up_pipeline_x4.set_progress_bar_config(disable=True)

    def __call__(self, image, prompt=''):
        with torch.no_grad():
            upscaled_image = self.up_pipeline_x4(
                prompt=[prompt],
                image=image,
                num_inference_steps=5,
            ).images[0]

        return upscaled_image
