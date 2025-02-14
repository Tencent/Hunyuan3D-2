from cog import BasePredictor, BaseModel, Input, Path
from torch import Generator
import os
from PIL import Image
import shutil
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline

MODEL_REPO = "tencent/Hunyuan3D-2"

class Output(BaseModel):
    mesh: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        os.environ["HY3DGEN_MODELS"] = "/src/checkpoints"
        self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(MODEL_REPO)
        self.texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(MODEL_REPO)
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()
        self.face_reduce_worker = FaceReducer()
        self.rmbg_worker = BackgroundRemover()

    def predict(
        self,
        image: Path = Input(
            description="Input image for generating 3D shape",
            default=None
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=20,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation",
            default=5.5,
            ge=1.0,
            le=20.0,
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=1234
        ),
        octree_resolution: int = Input(
            description="Octree resolution for mesh generation",
            choices=[256, 384, 512],
            default=256
        ),
        remove_background: bool = Input(
            description="Whether to remove background from input image",
            default=True
        ),
    ) -> Output:
        if os.path.exists("output"):
            shutil.rmtree("output")
        
        os.makedirs("output", exist_ok=True)

        max_facenum = 40000

        generator = Generator()
        generator = generator.manual_seed(seed)

        if image is not None:
            input_image = Image.open(str(image))
            if remove_background or input_image.mode == "RGB":
                input_image = self.rmbg_worker(input_image.convert('RGB'))
        else:
            raise ValueError("Image must be provided")

        input_image.save("output/input.png")

        mesh = self.i23d_worker(
            image=input_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution
        )[0]

        mesh = self.floater_remove_worker(mesh)
        mesh = self.degenerate_face_remove_worker(mesh)
        mesh = self.face_reduce_worker(mesh, max_facenum=max_facenum)
        mesh = self.texgen_worker(mesh, input_image)
        output_path = Path("output/mesh.glb")
        mesh.export(str(output_path), include_normals=True)

        if not Path(output_path).exists():
            raise RuntimeError(f"Failed to generate mesh file at {output_path}")

        return Output(mesh=output_path)