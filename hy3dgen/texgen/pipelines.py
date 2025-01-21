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


import logging
import os

import numpy as np
import torch
from PIL import Image

from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path):
        self.device = 'cuda'
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 1024
        self.bake_exp = 4
        self.merge_method = 'fast'


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    model_path = huggingface_hub.snapshot_download(repo_id=model_path)
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, 'hunyuan3d-paint-v2-0')
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))
                except ImportError:
                    logger.warning(
                        "You need to install HuggingFace Hub to load models from the hub."
                    )
                    raise RuntimeError(f"Model path {model_path} not found")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path))

        raise FileNotFoundError(f"Model path {original_model_path} not found and we could not find it at huggingface")

    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size)

        self.load_models()

    def load_models(self):
        # empty cude cache
        torch.cuda.empty_cache()
        # Load model
        self.models['delight_model'] = Light_Shadow_Remover(self.config)
        self.models['multiview_model'] = Multiview_Diffusion_Net(self.config)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    @torch.no_grad()
    def __call__(self, mesh, image):

        if isinstance(image, str):
            image_prompt = Image.open(image)
        else:
            image_prompt = image

        image_prompt = self.models['delight_model'](image_prompt)

        mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.models['multiview_model'](image_prompt, normal_maps + position_maps, camera_info)

        for i in range(len(multiviews)):
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        texture = self.texture_inpaint(texture, mask_np)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()

        return textured_mesh
