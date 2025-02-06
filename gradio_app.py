import os
import shutil
import time
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/*.png'))


def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def gen_save_folder(max_size=60):
    os.makedirs(SAVE_DIR, exist_ok=True)
    exists = set(int(_) for _ in os.listdir(SAVE_DIR) if _.isdigit())
    cur_id = min(set(range(max_size)) - exists) if len(exists) < max_size else -1
    if os.path.exists(f"{SAVE_DIR}/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"{SAVE_DIR}/{(cur_id + 1) % max_size}")
        print(f"remove {SAVE_DIR}/{(cur_id + 1) % max_size} success !!!")
    save_folder = f"{SAVE_DIR}/{max(0, cur_id)}"
    os.makedirs(save_folder, exist_ok=True)
    print(f"mkdir {save_folder} success !!!")
    return save_folder


def export_mesh(mesh, save_folder, textured=False):
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.glb')
    else:
        path = os.path.join(save_folder, f'white_mesh.glb')
    mesh.export(path, include_normals=textured)
    return path


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')

    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()
        obj_html = f"""
            <div class="column is-mobile is-centered">
                <model-viewer style="height: {height - 10}px; width: {width}px;" rotation-per-second="10deg" id="modelViewer"
                    src="{related_path}/" disable-tap 
                    environment-image="neutral" auto-rotate camera-target="0m 0m 0m" orientation="0deg 0deg 170deg" shadow-intensity=".9"
                    ar auto-rotate camera-controls>
                </model-viewer>
            </div>
            """

    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(template_html.replace('<model-viewer>', obj_html))

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" height="{height}" width="100%" frameborder="0"></iframe>'
    print(
        f'Find html file {output_html_path}, {os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """


def _gen_shape(
    caption,
    image,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    max_facenum = 40000,
):
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {}
    time_meta = {}
    start_time_0 = time.time()

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error(f"Text to 3D is disabled. Please enable it by restarted the app with `python gradio_app.py --enable_t23d`.")
        time_meta['text2image'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'input.png'))

    print(image.mode)
    if check_box_rembg or image.mode == "RGB":
        start_time = time.time()
        image = rmbg_worker(image.convert('RGB'))
        time_meta['rembg'] = time.time() - start_time

    image.save(os.path.join(save_folder, 'rembg.png'))

    # image to white model
    start_time = time.time()

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    mesh = i23d_worker(
        image=image,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        octree_resolution=octree_resolution
    )[0]

    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=max_facenum)

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    time_meta['image_to_textured_3d'] = {'total': time.time() - start_time}
    time_meta['total'] = time.time() - start_time_0
    stats['time'] = time_meta
    return mesh, image, save_folder


def generation_all(
    caption,
    image,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    max_facenum = 40000
):
    mesh, image, save_folder = _gen_shape(
        caption,
        image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        max_facenum=max_facenum
    )

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    textured_mesh = texgen_worker(mesh, image)
    path_textured = export_mesh(textured_mesh, save_folder, textured=True)
    model_viewer_html_textured = build_model_viewer_html(save_folder, height=596, width=700, textured=True)

    return (
        gr.update(value=path, visible=True),
        gr.update(value=path_textured, visible=True),
        model_viewer_html,
        model_viewer_html_textured,
    )


def shape_generation(
    caption,
    image,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    max_facenum = 40000
):
    mesh, image, save_folder = _gen_shape(
        caption,
        image,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        max_facenum=max_facenum
    )

    path = export_mesh(mesh, save_folder, textured=False)
    model_viewer_html = build_model_viewer_html(save_folder, height=596, width=700)

    return (
        gr.update(value=path, visible=True),
        model_viewer_html,
    )


def build_app():
    title_html = """
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">

    Hunyuan3D-2: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    <div align="center">
      <a href="https://github.com/tencent/Hunyuan3D-2">Github Page</a> &ensp; 
      <a href="http://3d-models.hunyuan.tencent.com">Homepage</a> &ensp;
      <a href="#">Technical Report</a> &ensp;
      <a href="https://huggingface.co/Tencent/Hunyuan3D-2"> Models</a> &ensp;
    </div>
    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.0') as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs() as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt') as tab_ip:
                        image = gr.Image(label='Image', type='pil', image_mode='RGBA', height=290)
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(value=True, label='Remove Background')

                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I) as tab_tp:
                        caption = gr.Textbox(label='Text Prompt',
                                             placeholder='HunyuanDiT will be used to generate image.',
                                             info='Example: A 3D model of a cute cat, white background')

                with gr.Accordion('Advanced Options', open=False):
                    num_steps = gr.Slider(maximum=100, minimum=1, value=30, step=1, label='Inference Steps')
                    octree_resolution = gr.Dropdown([256, 384, 512, 768, 1024], value=256, label='Octree Resolution')
                    cfg_scale = gr.Number(value=5.5, label='Guidance Scale')
                    max_facenum_slider = gr.Slider(maximum=200000, minimum=20000, value=40000, step=1000, label='Number of Faces')
                    seed = gr.Slider(maximum=1e7, minimum=0, value=1234, label='Seed')

                with gr.Group():
                    btn = gr.Button(value='Generate Shape Only', variant='primary')
                    btn_all = gr.Button(value='Generate Shape and Texture', variant='primary', visible=HAS_TEXTUREGEN)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab('Generated Mesh') as mesh1:
                        html_output1 = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Generated Textured Mesh') as mesh2:
                        html_output2 = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')

            with gr.Column(scale=2):
                with gr.Tabs() as gallery:
                    with gr.Tab('Image to 3D Gallery', id='tab_img_gallery') as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label="Image Prompts", examples_per_page=18)

                    with gr.Tab('Text to 3D Gallery', id='tab_txt_gallery', visible=HAS_T2I) as tab_gt:
                        with gr.Row():
                            gr.Examples(examples=example_ts, inputs=[caption],
                                        label="Text Prompts", examples_per_page=18)

        if not HAS_TEXTUREGEN:
            gr.HTML("""
            <div style="margin-top: 20px;">
                <b>Warning: </b>
                Texture synthesis is disabled due to missing requirements,
                 please refer to the README.md and install the missing requirements to activate it.
            </div>
            """)
        if not args.enable_t23d:
            gr.HTML("""
            <div style="margin-top: 20px;">
                <b>Warning: </b>
                Text to 3D is disabled. Please enable it by restarted the app with `python gradio_app.py --enable_t23d`.
            </div>
            """)

        tab_gi.select(fn=lambda: gr.update(selected='tab_img_prompt'), outputs=tabs_prompt)
        if HAS_T2I:
            tab_gt.select(fn=lambda: gr.update(selected='tab_txt_prompt'), outputs=tabs_prompt)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                max_facenum_slider
            ],
            outputs=[file_out, html_output1]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[file_out],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                max_facenum_slider
            ],
            outputs=[file_out, file_out2, html_output1, html_output2]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[file_out, file_out2],
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--cache-path', type=str, default='gradio_cache')
    parser.add_argument('--enable_t23d', action='store_true')
    args = parser.parse_args()

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

    HTML_OUTPUT_PLACEHOLDER = """
    <div style='height: 596px; width: 100%; border-radius: 8px; border-color: #e5e7eb; order-style: solid; border-width: 1px;'></div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    try:
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        texgen_worker = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        HAS_TEXTUREGEN = True
    except Exception as e:
        print(e)
        print("Failed to load texture generator.")
        print('Please refer to the README.md and install the missing requirements to activate it.')
        HAS_TEXTUREGEN = False

    HAS_T2I = False
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True

    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, \
        Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()
    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port)
