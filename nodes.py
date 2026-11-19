import base64
import io
import os
import re

import numpy as np
import torch
from PIL import Image

MODELS = ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]
QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
BACKGROUNDS = ["auto", "opaque", "transparent"]
OUTPUT_FORMATS = ["png", "jpeg", "webp"]
MAX_REFERENCE_IMAGES = 5
SIZE_PATTERN = re.compile(r"^(\d+)x(\d+)$")


def validate_size(size):
    """Normalise the size input to "auto" or "WIDTHxHEIGHT" with multiples of 16."""
    size = size.strip().lower()
    if size == "auto":
        return size
    match = SIZE_PATTERN.match(size)
    if not match:
        raise ValueError(f'Invalid size "{size}". Use "auto" or WIDTHxHEIGHT, e.g. 1024x1024.')
    width, height = (int(v) for v in match.groups())
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError(f"Invalid size {size}: width and height must be positive multiples of 16.")
    return size


def tensor_to_png_files(image, prefix):
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] (0-1 floats) into PNG upload tuples."""
    files = []
    for i, frame in enumerate(image):
        array = np.clip(frame.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        files.append((f"{prefix}_{i}.png", buffer.getvalue(), "image/png"))
    return files


def b64_to_tensors(b64_data):
    """Decode a base64 image from the API into ComfyUI IMAGE [1, H, W, 3] and MASK [1, H, W] tensors.

    The mask follows ComfyUI's LoadImage convention: 1 where the image is transparent.
    """
    image = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGBA")
    array = np.asarray(image).astype(np.float32) / 255.0
    rgb = torch.from_numpy(array[..., :3])[None,]
    mask = 1.0 - torch.from_numpy(array[..., 3])[None,]
    return rgb, mask


class OpenAIImageGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"reference_image_{i}": ("IMAGE",) for i in range(1, MAX_REFERENCE_IMAGES + 1)
        }
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "OpenAI API key. Leave empty to use the OPENAI_API_KEY environment variable.",
                }),
                "model": (MODELS, {"default": MODELS[0]}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "size": ("STRING", {
                    "default": "auto",
                    "tooltip": 'Output size as WIDTHxHEIGHT (multiples of 16, e.g. 1536x1024) or "auto".',
                }),
                "quality": (QUALITIES, {"default": "auto"}),
                "background": (BACKGROUNDS, {
                    "default": "auto",
                    "tooltip": "Transparent backgrounds require png or webp output.",
                }),
                "output_format": (OUTPUT_FORMATS, {"default": "png"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "generate"
    CATEGORY = "OpenAI"

    def generate(self, api_key, model, prompt, size="auto", quality="auto", background="auto",
                 output_format="png", **reference_images):
        from openai import OpenAI

        api_key = api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("No OpenAI API key provided (set the api_key input or OPENAI_API_KEY).")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        size = validate_size(size)
        if background == "transparent" and output_format == "jpeg":
            raise ValueError("Transparent backgrounds require png or webp output, not jpeg.")

        client = OpenAI(api_key=api_key)

        files = []
        for i in range(1, MAX_REFERENCE_IMAGES + 1):
            image = reference_images.get(f"reference_image_{i}")
            if image is not None:
                files.extend(tensor_to_png_files(image, f"reference_{i}"))

        params = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
        if files:
            result = client.images.edit(image=files, **params)
        else:
            result = client.images.generate(**params)

        images, masks = zip(*(b64_to_tensors(item.b64_json) for item in result.data))
        return (torch.cat(images, dim=0), torch.cat(masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "OpenAIImageGeneration": OpenAIImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIImageGeneration": "OpenAI Image Generation",
}
