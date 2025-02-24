[中文阅读](README_zh_cn.md)
[日本語で読む](README_ja_jp.md)

<p align="center"> 
  <img src="./assets/images/teaser.jpg">

</p>

<div align="center">
  <a href=https://3d.hunyuan.tencent.com target="_blank"><img src=https://img.shields.io/badge/Official%20Site-333399.svg?logo=homepage height=22px></a>
  <a href=https://huggingface.co/spaces/tencent/Hunyuan3D-2  target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/tencent/Hunyuan3D-2 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://3d-models.hunyuan.tencent.com/ target="_blank"><img src= https://img.shields.io/badge/Page-bb8a2e.svg?logo=github height=22px></a>
  <a href=https://discord.gg/dNBrdrGGMa target="_blank"><img src= https://img.shields.io/badge/Discord-white.svg?logo=discord height=22px></a>
  <a href=https://arxiv.org/abs/2501.12202 target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>
  <a href=https://x.com/txhunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
 <a href="#community-resources" target="_blank"><img src=https://img.shields.io/badge/Community-lavender.svg?logo=homeassistantcommunitystore height=22px></a>
</div>

[//]: # (  <a href=# target="_blank"><img src=https://img.shields.io/badge/Report-b5212f.svg?logo=arxiv height=22px></a>)

[//]: # (  <a href=# target="_blank"><img src= https://img.shields.io/badge/Colab-8f2628.svg?logo=googlecolab height=22px></a>)

[//]: # (  <a href="#"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/v/mulankit?logo=pypi"  height=22px></a>)

<br>


> Join our **[Wechat](#)** and **[Discord](https://discord.gg/dNBrdrGGMa)** group to discuss and find help from us.

| Wechat Group                                     | Xiaohongshu                                           | X                                           | Discord                                           |
|--------------------------------------------------|-------------------------------------------------------|---------------------------------------------|---------------------------------------------------|
| <img src="assets/qrcode/wechat.png"  height=140> | <img src="assets/qrcode/xiaohongshu.png"  height=140> | <img src="assets/qrcode/x.png"  height=140> | <img src="assets/qrcode/discord.png"  height=140> |        

---

<p align="center">
“ Living out everyone’s imagination on creating and manipulating 3D assets.”
</p>

## 🔥 News

- Feb 14, 2025: 🛠️ Release texture enhancement module, please obtain high-definition textures via [here](minimal_demo.py)!
- Feb 3, 2025: 🐎
  Release [Hunyuan3D-DiT-v2-0-Fast](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-fast), our
  guidance distillation model that could half the dit inference time, see [here](minimal_demo.py) for usage.
- Jan 27, 2025: 🛠️ Release Blender addon for Hunyuan3D 2.0, Check it out [here](#blender-addon).
- Jan 23, 2025: 💬 We thank community members for
  creating [Windows installation tool](https://github.com/YanWenKun/Hunyuan3D-2-WinPortable), ComfyUI support
  with [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
  and [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack) and other
  awesome [extensions](#community-resources).
- Jan 21, 2025: 💬 Enjoy exciting 3D generation on our website [Hunyuan3D Studio](https://3d.hunyuan.tencent.com)!
- Jan 21, 2025: 🤗 Release inference code and pretrained models
  of [Hunyuan3D 2.0](https://huggingface.co/tencent/Hunyuan3D-2). Please give it a try
  via [huggingface space](https://huggingface.co/spaces/tencent/Hunyuan3D-2) and
  our [official site](https://3d.hunyuan.tencent.com)!

## **Abstract**

We present Hunyuan3D 2.0, an advanced large-scale 3D synthesis system for generating high-resolution textured 3D assets.
This system includes two foundation components: a large-scale shape generation model - Hunyuan3D-DiT, and a large-scale
texture synthesis model - Hunyuan3D-Paint.
The shape generative model, built on a scalable flow-based diffusion transformer, aims to create geometry that properly
aligns with a given condition image, laying a solid foundation for downstream applications.
The texture synthesis model, benefiting from strong geometric and diffusion priors, produces high-resolution and vibrant
texture maps for either generated or hand-crafted meshes.
Furthermore, we build Hunyuan3D-Studio - a versatile, user-friendly production platform that simplifies the re-creation
process of 3D assets. It allows both professional and amateur users to manipulate or even animate their meshes
efficiently.
We systematically evaluate our models, showing that Hunyuan3D 2.0 outperforms previous state-of-the-art models,
including the open-source models and closed-source models in geometry details, condition alignment, texture quality, and
e.t.c.



<p align="center">
  <img src="assets/images/system.jpg">
</p>

## ☯️ **Hunyuan3D 2.0**

### Architecture

Hunyuan3D 2.0 features a two-stage generation pipeline, starting with the creation of a bare mesh, followed by the
synthesis of a texture map for that mesh. This strategy is effective for decoupling the difficulties of shape and
texture generation and also provides flexibility for texturing either generated or handcrafted meshes.

<p align="left">
  <img src="assets/images/arch.jpg">
</p>

### Performance

We have evaluated Hunyuan3D 2.0 with other open-source as well as close-source 3d-generation methods.
The numerical results indicate that Hunyuan3D 2.0 surpasses all baselines in the quality of generated textured 3D assets
and the condition following ability.

| Model                   | CMMD(⬇)   | FID_CLIP(⬇) | FID(⬇)      | CLIP-score(⬆) |
|-------------------------|-----------|-------------|-------------|---------------|
| Top Open-source Model1  | 3.591     | 54.639      | 289.287     | 0.787         |
| Top Close-source Model1 | 3.600     | 55.866      | 305.922     | 0.779         |
| Top Close-source Model2 | 3.368     | 49.744      | 294.628     | 0.806         |
| Top Close-source Model3 | 3.218     | 51.574      | 295.691     | 0.799         |
| Hunyuan3D 2.0           | **3.193** | **49.165**  | **282.429** | **0.809**     |

Generation results of Hunyuan3D 2.0:
<p align="left">
  <img src="assets/images/e2e-1.gif"  height=250>
  <img src="assets/images/e2e-2.gif"  height=250>
</p>

## 🎁 Models Zoo

It takes 11.5 GB VRAM for shape generation and 24.5 GB for shape and texture generation in total.

| Model                   |    Description                         | Date       | Size | Huggingface                                                                              |
|-------------------------|-----------------------------|------------|------|------------------------------------------------------------------------------------------| 
| Hunyuan3D-DiT-v2-0-Fast | Guidance Distillation Model | 2025-02-03 | 2.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0-fast) |
| Hunyuan3D-DiT-v2-0      | Image to Shape Model        | 2025-01-21 | 2.6B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-dit-v2-0)      |
| Hunyuan3D-Paint-v2-0    | Texture Generation Model    | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-paint-v2-0)    |
| Hunyuan3D-Delight-v2-0  | Image Delight Model         | 2025-01-21 | 1.3B | [Download](https://huggingface.co/tencent/Hunyuan3D-2/tree/main/hunyuan3d-delight-v2-0)  | 

## 🤗 Get Started with Hunyuan3D 2.0

You may follow the next steps to use Hunyuan3D 2.0 via:

- [Code](#code-usage)
- [Gradio App](#gradio-app)
- [API Server](#api-server)
- [Blender Addon](#blender-addon)
- [Official Site](#official-site)

### Install Requirements

Please install Pytorch via the [official](https://pytorch.org/) site. Then install the other requirements via

```bash
pip install -r requirements.txt
# for texture
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install
```

### Code Usage

We designed a diffusers-like API to use our shape generation model - Hunyuan3D-DiT and texture synthesis model -
Hunyuan3D-Paint.

You could assess **Hunyuan3D-DiT** via:

```python
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]
```

The output mesh is a [trimesh object](https://trimesh.org/trimesh.html), which you could save to glb/obj (or other
format) file.

For **Hunyuan3D-Paint**, do the following:

```python
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# let's generate a mesh first
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(image='assets/demo.png')[0]

pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
mesh = pipeline(mesh, image='assets/demo.png')
```

Please visit [minimal_demo.py](minimal_demo.py) for more advanced usage, such as **text to 3D** and **texture generation
for handcrafted mesh**.

### Gradio App

You could also host a [Gradio](https://www.gradio.app/) App in your own computer via:

```bash
python3 gradio_app.py
```

### API Server

You could launch an API server locally, which you could post web request for Image/Text to 3D, Texturing existing mesh,
and e.t.c.

```bash
python api_server.py --host 0.0.0.0 --port 8080
```

A demo post request for image to 3D without texture.

```bash
img_b64_str=$(base64 -i assets/demo.png)
curl -X POST "http://localhost:8080/generate" \
     -H "Content-Type: application/json" \
     -d '{
           "image": "'"$img_b64_str"'",
         }' \
     -o test2.glb
```

### Blender Addon

With an API server launched, you could also directly use Hunyuan3D 2.0 in your blender with
our [Blender Addon](blender_addon.py). Please follow our tutorial to install and use.

https://github.com/user-attachments/assets/8230bfb5-32b1-4e48-91f4-a977c54a4f3e

### Official Site

Don't forget to visit [Hunyuan3D](https://3d.hunyuan.tencent.com) for quick use, if you don't want to host yourself.

## 📑 Open-Source Plan

- [x] Inference Code
- [x] Model Checkpoints
- [x] Technical Report
- [ ] ComfyUI
- [ ] TensorRT Version

## 🔗 BibTeX

If you found this repository helpful, please cite our reports:

```bibtex
@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{yang2024hunyuan3d,
    title={Hunyuan3D 1.0: A Unified Framework for Text-to-3D and Image-to-3D Generation},
    author={Tencent Hunyuan3D Team},
    year={2024},
    eprint={2411.02293},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Community Resources

Thanks for the contributions of community members, here we have these great extensions of Hunyuan3D 2.0:

- [ComfyUI-3D-Pack](https://github.com/MrForExample/ComfyUI-3D-Pack)
- [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper)
- [Hunyuan3D-2-for-windows](https://github.com/sdbds/Hunyuan3D-2-for-windows)
- [📦 A bundle for running on Windows | 整合包](https://github.com/YanWenKun/Hunyuan3D-2-WinPortable)
- [Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)
- [Kaggle Notebook](https://github.com/darkon12/Hunyuan3D-2GP_Kaggle)

## Acknowledgements

We would like to thank the contributors to
the [DINOv2](https://github.com/facebookresearch/dinov2), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers), [HuggingFace](https://huggingface.co), [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D),
and [Michelangelo](https://github.com/NeuralCarver/Michelangelo/tree/main) repositories, for their open research and
exploration.

## Star History

<a href="https://star-history.com/#Tencent/Hunyuan3D-2&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent/Hunyuan3D-2&type=Date" />
 </picture>
</a>
