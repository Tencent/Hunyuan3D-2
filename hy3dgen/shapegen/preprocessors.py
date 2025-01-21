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
from einops import repeat, rearrange


def array_to_tensor(np_array):
    image_pt = torch.tensor(np_array).float()
    image_pt = image_pt / 255 * 2 - 1
    image_pt = rearrange(image_pt, "h w c -> c h w")
    image_pts = repeat(image_pt, "c h w -> b c h w", b=1)
    return image_pts


class ImageProcessorV2:
    def __init__(self, size=512, border_ratio=None):
        self.size = size
        self.border_ratio = border_ratio

    @staticmethod
    def recenter(image, border_ratio: float = 0.2):
        """ recenter an image to leave some empty space at the image border.

        Args:
            image (ndarray): input image, float/uint8 [H, W, 3/4]
            mask (ndarray): alpha mask, bool [H, W]
            border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

        Returns:
            ndarray: output image, float/uint8 [H, W, 3/4]
        """

        if image.shape[-1] == 4:
            mask = image[..., 3]
        else:
            mask = np.ones_like(image[..., 0:1]) * 255
            image = np.concatenate([image, mask], axis=-1)
            mask = mask[..., 0]

        H, W, C = image.shape

        size = max(H, W)
        result = np.zeros((size, size, C), dtype=np.uint8)

        coords = np.nonzero(mask)
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        h = x_max - x_min
        w = y_max - y_min
        if h == 0 or w == 0:
            raise ValueError('input image is empty')
        desired_size = int(size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (size - h2) // 2
        x2_max = x2_min + h2

        y2_min = (size - w2) // 2
        y2_max = y2_min + w2

        result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2),
                                                          interpolation=cv2.INTER_AREA)

        bg = np.ones((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255
        # bg = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8) * 255
        mask = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3] * mask + bg * (1 - mask)

        mask = mask * 255
        result = result.clip(0, 255).astype(np.uint8)
        mask = mask.clip(0, 255).astype(np.uint8)
        return result, mask

    def __call__(self, image, border_ratio=0.15, to_tensor=True, return_mask=False, **kwargs):
        if self.border_ratio is not None:
            border_ratio = self.border_ratio
            print(f"Using border_ratio from init: {border_ratio}")
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image, mask = self.recenter(image, border_ratio=border_ratio)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.asarray(image)
            image, mask = self.recenter(image, border_ratio=border_ratio)

        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = mask[..., np.newaxis]

        if to_tensor:
            image = array_to_tensor(image)
            mask = array_to_tensor(mask)
        if return_mask:
            return image, mask
        return image


IMAGE_PROCESSORS = {
    "v2": ImageProcessorV2,
}

DEFAULT_IMAGEPROCESSOR = 'v2'
