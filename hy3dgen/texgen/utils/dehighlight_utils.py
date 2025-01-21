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

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


class Light_Shadow_Remover():
    def __init__(self, config):
        self.device = config.device
        self.cfg_image = 1.5
        self.cfg_text = 1.0

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            config.light_remover_ckpt_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline.set_progress_bar_config(disable=True)

        self.pipeline = pipeline.to(self.device, torch.float16)

    @torch.no_grad()
    def __call__(self, image):

        image = image.resize((512, 512))

        if image.mode == 'RGBA':
            image_array = np.array(image)
            alpha_channel = image_array[:, :, 3]
            erosion_size = 3
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)
            image_array[alpha_channel == 0, :3] = 255
            image_array[:, :, 3] = alpha_channel
            image = Image.fromarray(image_array)

            image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
            alpha = image_tensor[:, :, 3:]
            rgb_target = image_tensor[:, :, :3]
        else:
            image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
            alpha = torch.ones_like(image_tensor)[:, :, :1]
            rgb_target = image_tensor[:, :, :3]

        image = image.convert('RGB')

        image = self.pipeline(
            prompt="",
            image=image,
            generator=torch.manual_seed(42),
            height=512,
            width=512,
            num_inference_steps=50,
            image_guidance_scale=self.cfg_image,
            guidance_scale=self.cfg_text,
        ).images[0]

        return image
