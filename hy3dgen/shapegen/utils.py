import logging
import os
from functools import wraps

import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger('hy3dgen.shapgen')


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.

        Supports both context manager and decorator usage.

        Example as context manager:
        ```python
        with synchronize_timer('name') as t:
            run()
        ```

        Example as decorator:
        ```python
        @synchronize_timer('Export to trimesh')
        def export_to_trimesh(mesh_output):
            pass
        ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f'{self.name} takes {self.time} ms')

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    # try local path
    base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
    model_path = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))
    logger.info(f'Try to load model from local path: {model_path}')
    if not os.path.exists(model_path):
        logger.info('Model path not exists, try to download from huggingface')
        try:
            import huggingface_hub
            # download from huggingface
            path = huggingface_hub.snapshot_download(repo_id=original_model_path)
            model_path = os.path.join(path, subfolder)
        except ImportError:
            logger.warning(
                "You need to install HuggingFace Hub to load models from the hub."
            )
            raise RuntimeError(f"Model path {model_path} not found")
        except Exception as e:
            raise e

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {original_model_path} not found")

    extension = 'ckpt' if not use_safetensors else 'safetensors'
    variant = '' if variant is None else f'.{variant}'
    ckpt_name = f'model{variant}.{extension}'
    config_path = os.path.join(model_path, 'config.yaml')
    ckpt_path = os.path.join(model_path, ckpt_name)
    return config_path, ckpt_path
