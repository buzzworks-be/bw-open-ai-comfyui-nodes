# OpenAI ComfyUI Nodes

A ComfyUI custom node that generates images with OpenAI's GPT Image 2.5 models.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> openai-comfyui-nodes
pip install -r openai-comfyui-nodes/requirements.txt
```

Restart ComfyUI. The node appears under **OpenAI → OpenAI Image Generation**.

## Inputs

| Input | Type | Notes |
| --- | --- | --- |
| `api_key` | string | Your OpenAI API key. If empty, `OPENAI_API_KEY` from the environment is used. |
| `model` | choice | `gpt-image-2.5-flare` (faster) or `gpt-image-2.5-sunburst` (stronger editing / multi-image fusion). |
| `prompt` | string | What to generate. |
| `size` | string | `auto` or `WIDTHxHEIGHT`, both multiples of 16 (e.g. `1024x1024`, `1536x1024`). |
| `quality` | choice | `auto`, `low`, `medium`, `high`, `xhigh`, `max`. |
| `background` | choice | `auto`, `opaque`, `transparent`. Transparent requires `png` or `webp`. |
| `output_format` | choice | `png` (default), `jpeg`, `webp`. |
| `reference_image_1` … `reference_image_5` | image (optional) | Reference images. |

With no reference images the node calls `images.generate`; with one or more it calls `images.edit`, sending all references together.

Outputs:
- `image` (`IMAGE`): the generated image(s).
- `mask` (`MASK`): the alpha channel, using the same convention as ComfyUI's Load Image node (1 = transparent). All zeros for opaque images.

> **Note:** a key typed into `api_key` is saved inside the workflow JSON (and in PNG metadata of saved images). Use the `OPENAI_API_KEY` environment variable if you plan to share workflows or images.

## Tests

```bash
pip install pytest torch pillow numpy openai
cd tests && pytest
```
