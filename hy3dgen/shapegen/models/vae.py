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

from typing import Tuple, List, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from skimage import measure
from tqdm import tqdm


class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class MLP(nn.Module):
    def __init__(
        self, *,
        width: int,
        output_width: int = None,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, output_width if output_width is not None else width)
        self.gelu = nn.GELU()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_data: Optional[int] = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)

        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(width // heads, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary"
    ):
        super().__init__()

        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        occ = self.output_proj(x)
        return occ


def generate_dense_grid_points(bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               octree_depth: int,
                               indexing: str = "ij",
                               octree_resolution: int = None,
                               ):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_depth)
    if octree_resolution is not None:
        num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class Latent2MeshOutput:

    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


class ShapeVAE(nn.Module):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
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
            width=width,
            heads=heads,
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

    @torch.no_grad()
    def latents2mesh(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = 1.1,
        octree_depth: int = 7,
        num_chunks: int = 10000,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        mc_algo: str = 'dmc',
    ):
        device = latents.device

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)

        # 2. latents to 3d volume
        batch_logits = []
        batch_size = latents.shape[0]
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks),
                          desc=f"MC Level {mc_level} Implicit Function:"):
            queries = xyz_samples[start: start + num_chunks, :].to(device)
            queries = queries.half()
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = self.geo_decoder(batch_queries.to(latents.dtype), latents)
            if mc_level == -1:
                mc_level = 0
                logits = torch.sigmoid(logits) * 2 - 1
                print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float()

        # 3. extract surface
        outputs = []
        for i in range(batch_size):
            try:
                if mc_algo == 'mc':
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_logits[i].cpu().numpy(),
                        mc_level,
                        method="lewiner"
                    )
                    vertices = vertices / grid_size * bbox_size + bbox_min
                elif mc_algo == 'dmc':
                    if not hasattr(self, 'dmc'):
                        try:
                            from diso import DiffDMC
                        except:
                            raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
                        self.dmc = DiffDMC(dtype=torch.float32).to(device)
                    octree_resolution = 2 ** octree_depth if octree_resolution is None else octree_resolution
                    sdf = -grid_logits[i] / octree_resolution
                    verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
                    verts = center_vertices(verts)
                    vertices = verts.detach().cpu().numpy()
                    faces = faces.detach().cpu().numpy()[:, ::-1]
                else:
                    raise ValueError(f"mc_algo {mc_algo} not supported.")

                outputs.append(
                    Latent2MeshOutput(
                        mesh_v=vertices.astype(np.float32),
                        mesh_f=np.ascontiguousarray(faces)
                    )
                )

            except ValueError:
                outputs.append(None)
            except RuntimeError:
                outputs.append(None)

        return outputs
