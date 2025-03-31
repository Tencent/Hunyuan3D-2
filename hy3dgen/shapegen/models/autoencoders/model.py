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


import os

import torch
import torch.nn as nn
import yaml

from .attention_blocks import FourierEmbedder, Transformer, CrossAttentionDecoder
from .surface_extractors import MCSurfaceExtractor, SurfaceExtractors
from .volume_decoders import VanillaVolumeDecoder, FlashVDMVolumeDecoding, HierarchicalVolumeDecoding
from ...utils import logger, synchronize_timer, smart_load_model


class VectsetVAE(nn.Module):

    @classmethod
    @synchronize_timer('VectsetVAE Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")

        logger.info(f"Loading model from {ckpt_path}")
        if use_safetensors:
            import safetensors.torch
            ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        model_kwargs = config['params']
        model_kwargs.update(kwargs)

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt)
        model.to(device=device, dtype=dtype)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
        subfolder='hunyuan3d-vae-v2-0',
        **kwargs,
    ):
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )

        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        volume_decoder=None,
        surface_extractor=None
    ):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        with synchronize_timer('Volume decoding'):
            grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        with synchronize_timer('Surface extraction'):
            outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs

    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='dmc',
    ):
        if enabled:
            if adaptive_kv_selection:
                self.volume_decoder = FlashVDMVolumeDecoding(topk_mode)
            else:
                self.volume_decoder = HierarchicalVolumeDecoding()
            if mc_algo not in SurfaceExtractors.keys():
                raise ValueError(f'Unsupported mc_algo {mc_algo}, available: {list(SurfaceExtractors.keys())}')
            self.surface_extractor = SurfaceExtractors[mc_algo]()
        else:
            self.volume_decoder = VanillaVolumeDecoder()
            self.surface_extractor = MCSurfaceExtractor()


class ShapeVAE(VectsetVAE):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.geo_decoder_ln_post = geo_decoder_ln_post

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents
