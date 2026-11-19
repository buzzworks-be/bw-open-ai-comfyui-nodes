import base64
import io
import os
import re

import numpy as np
import torch
import requests
from PIL import Image

OPENROUTER_IMAGES_URL = os.environ.get("OPENROUTER_IMAGES_URL", "https://openrouter.ai/api/v1/images")
REQUEST_TIMEOUT = 600
PROVIDERS = ["auto", "openrouter", "openai"]
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


def tensor_to_data_urls(image):
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] (0-1 floats) into base64 PNG data URLs."""
    urls = []
    for frame in image:
        array = np.clip(frame.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    return urls


def tensor_to_png_files(image, prefix):
    """Convert a ComfyUI IMAGE tensor into PNG upload tuples (OpenAI SDK format)."""
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


def resolve_provider(provider, openrouter_key, openai_key):
    """Pick the provider and key. "auto" prefers OpenRouter, then OpenAI.

    Keys come from the node inputs first, then OPENROUTER_API_KEY / OPENAI_API_KEY.
    """
    openrouter_key = openrouter_key.strip() or os.environ.get("OPENROUTER_API_KEY", "")
    openai_key = openai_key.strip() or os.environ.get("OPENAI_API_KEY", "")
    if provider == "auto":
        if openrouter_key:
            provider = "openrouter"
        elif openai_key:
            provider = "openai"
        else:
            raise ValueError(
                "No API key found. Fill in openrouter_api_key or openai_api_key "
                "(or set OPENROUTER_API_KEY / OPENAI_API_KEY)."
            )
    key = openrouter_key if provider == "openrouter" else openai_key
    if not key:
        field = "openrouter_api_key" if provider == "openrouter" else "openai_api_key"
        raise ValueError(f'Provider "{provider}" selected but no key given. Fill in {field}.')
    return provider, key


def generate_openrouter(api_key, model, params, reference_images):
    references = []
    for image in reference_images:
        references.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in tensor_to_data_urls(image)
        )
    payload = {"model": f"openai/{model}", **params}
    if references:
        payload["input_references"] = references

    response = requests.post(
        OPENROUTER_IMAGES_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {response.text}")
    data = response.json().get("data") or []
    if not data:
        raise RuntimeError(f"OpenRouter returned no images: {response.text[:500]}")
    return [item["b64_json"] for item in data]


def generate_openai(api_key, model, params, reference_images):
    from openai import OpenAI  # only needed for the direct OpenAI provider

    client = OpenAI(api_key=api_key)
    files = []
    for i, image in enumerate(reference_images, start=1):
        files.extend(tensor_to_png_files(image, f"reference_{i}"))
    if files:
        result = client.images.edit(model=model, image=files, **params)
    else:
        result = client.images.generate(model=model, **params)
    return [item.b64_json for item in result.data]


class OpenAIImageGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            f"reference_image_{i}": ("IMAGE",) for i in range(1, MAX_REFERENCE_IMAGES + 1)
        }
        return {
            "required": {
                "provider": (PROVIDERS, {
                    "default": "auto",
                    "tooltip": "auto: use OpenRouter if an OpenRouter key is available, otherwise OpenAI. "
                               "Or force one.",
                }),
                "openrouter_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "OpenRouter key. Leave empty to use the OPENROUTER_API_KEY environment variable.",
                }),
                "openai_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "OpenAI key. Leave empty to use the OPENAI_API_KEY environment variable.",
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
    CATEGORY = "OpenAI/OpenRouter"

    def generate(self, provider, openrouter_api_key, openai_api_key, model, prompt, size="auto",
                 quality="auto", background="auto", output_format="png", **reference_images):
        provider, api_key = resolve_provider(provider, openrouter_api_key, openai_api_key)
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        size = validate_size(size)
        if background == "transparent" and output_format == "jpeg":
            raise ValueError("Transparent backgrounds require png or webp output, not jpeg.")

        refs = [
            reference_images[f"reference_image_{i}"]
            for i in range(1, MAX_REFERENCE_IMAGES + 1)
            if reference_images.get(f"reference_image_{i}") is not None
        ]
        params = {
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "background": background,
            "output_format": output_format,
        }
        generate = generate_openrouter if provider == "openrouter" else generate_openai
        b64_images = generate(api_key, model, params, refs)

        images, masks = zip(*(b64_to_tensors(b) for b in b64_images))
        return (torch.cat(images, dim=0), torch.cat(masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "BWOpenRouterImageGeneration": OpenAIImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BWOpenRouterImageGeneration": "GPT Image (OpenRouter / OpenAI)",
}
