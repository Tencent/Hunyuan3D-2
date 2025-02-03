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
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline


def image_to_3d(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('mesh.glb')

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        mesh = pipeline(mesh, image=image)
        mesh.export('texture.glb')
    except Exception as e:
        print(e)
        print('Please try to install requirements by following README.md')


def text_to_3d(prompt='a car'):
    rembg = BackgroundRemover()
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
    model_path = 'tencent/Hunyuan3D-2'
    i23d = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    image = t2i(prompt)
    image = rembg(image)
    mesh = i23d(image, num_inference_steps=30, mc_algo='mc')[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('t2i_demo.glb')


def image_to_3d_fast(image_path='assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

    image = Image.open(image_path)

    if image.mode == 'RGB':
        image = rembg(image)

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-dit-v2-0-fast',
        variant='fp16'
    )

    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh)
    mesh.export('mesh.glb')


if __name__ == '__main__':
    image_to_3d_fast()
    # image_to_3d()
    # text_to_3d()
