# OpenAI ComfyUI Nodes

A ComfyUI custom node that generates images with OpenAI's GPT Image 2.5 models, either directly through OpenAI or through [OpenRouter](https://openrouter.ai).

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> openai-comfyui-nodes
pip install -r openai-comfyui-nodes/requirements.txt
```

Restart ComfyUI. The node appears under **OpenAI → OpenRouter → GPT Image (OpenRouter / OpenAI)**.

## Choosing OpenRouter or OpenAI

You only need **one** key. Fill in the field for the service you want (or set the matching environment variable):

| Service | Node field | Environment variable |
| --- | --- | --- |
| OpenRouter | `openrouter_api_key` | `OPENROUTER_API_KEY` |
| OpenAI (direct) | `openai_api_key` | `OPENAI_API_KEY` |

The `provider` input decides which one is used:

- `auto` (default): uses **OpenRouter** if an OpenRouter key is available, otherwise **OpenAI**. If you fill in both keys, OpenRouter wins.
- `openrouter` / `openai`: always use that service (and error clearly if its key is missing).

A key typed into the node always takes priority over the environment variable.

## Inputs

| Input | Type | Notes |
| --- | --- | --- |
| `provider` | choice | `auto`, `openrouter`, `openai` (see above). |
| `openrouter_api_key` | string | OpenRouter key. If empty, `OPENROUTER_API_KEY` is used. |
| `openai_api_key` | string | OpenAI key. If empty, `OPENAI_API_KEY` is used. |
| `model` | choice | `gpt-image-2.5-flare` (faster) or `gpt-image-2.5-sunburst` (stronger editing / multi-image fusion). Sent as `openai/<model>` on OpenRouter. |
| `prompt` | string | What to generate. |
| `size` | string | `auto` or `WIDTHxHEIGHT`, both multiples of 16 (e.g. `1024x1024`, `1536x1024`). |
| `quality` | choice | `auto`, `low`, `medium`, `high`, `xhigh`, `max`. |
| `background` | choice | `auto`, `opaque`, `transparent`. Transparent requires `png` or `webp`. |
| `output_format` | choice | `png` (default), `jpeg`, `webp`. |
| `reference_image_1` … `reference_image_5` | image (optional) | Reference images. |

Under the hood:
- **OpenRouter:** `POST https://openrouter.ai/api/v1/images`; references are sent together as `input_references`.
- **OpenAI:** `images.generate` with no references, `images.edit` (all references together) with one or more. The `openai` package is only imported when this provider is used.

Outputs:
- `image` (`IMAGE`): the generated image(s).
- `mask` (`MASK`): the alpha channel, using the same convention as ComfyUI's Load Image node (1 = transparent). All zeros for opaque images.

> **Note:** a key typed into the node is saved inside the workflow JSON (and in PNG metadata of saved images). Use the environment variables if you plan to share workflows or images.

> **Upgrading:** the old single `api_key` input was replaced by `openrouter_api_key` and `openai_api_key`. Re-enter your key in existing workflows.

## Tests

```bash
pip install pytest torch pillow numpy requests openai
cd tests && pytest
```

## Trying it without a real key

`python tests/mock_openrouter.py` starts a local fake OpenRouter endpoint; point the node at it with
`OPENROUTER_IMAGES_URL=http://127.0.0.1:8765/api/v1/images` when starting ComfyUI.

## Using alongside other OpenAI node packs

The node's internal ID is `BWOpenRouterImageGeneration` (ComfyUI keeps one global list of node IDs, so it must not match another pack's). It has its own display name and menu category, so it sits next to the original ComfyUI-OpenAI nodes without clashing. Only the folder needs a unique name.
