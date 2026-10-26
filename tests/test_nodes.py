import base64
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nodes  # noqa: E402


import pytest  # noqa: E402

DEFAULTS = {"size": "auto", "quality": "auto", "background": "auto", "output_format": "png"}


def fake_b64(w=8, h=4, alpha=255):
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), (255, 0, 0, alpha)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run(alpha=255, **kwargs):
    client = mock.MagicMock()
    response = SimpleNamespace(data=[SimpleNamespace(b64_json=fake_b64(alpha=alpha))])
    client.images.generate.return_value = response
    client.images.edit.return_value = response
    with mock.patch("openai.OpenAI", return_value=client) as ctor:
        out, mask = nodes.OpenAIImageGeneration().generate(**kwargs)
    return out, mask, client, ctor


def test_generate_without_references():
    out, mask, client, ctor = run(api_key="sk-test", model="gpt-image-2.5-flare", prompt="a cat")
    ctor.assert_called_once_with(api_key="sk-test")
    client.images.generate.assert_called_once_with(
        model="gpt-image-2.5-flare", prompt="a cat", **DEFAULTS
    )
    client.images.edit.assert_not_called()
    assert out.shape == (1, 4, 8, 3)
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert mask.shape == (1, 4, 8)
    assert torch.all(mask == 0)


def test_generation_options_are_forwarded():
    _, mask, client, _ = run(
        alpha=0, api_key="sk-test", model="gpt-image-2.5-flare", prompt="logo",
        size="1536X1024", quality="high", background="transparent", output_format="webp",
    )
    client.images.generate.assert_called_once_with(
        model="gpt-image-2.5-flare", prompt="logo", size="1536x1024", quality="high",
        background="transparent", output_format="webp",
    )
    assert torch.all(mask == 1)


@pytest.mark.parametrize("size", ["1024", "1000x1024", "0x1024", "big"])
def test_invalid_size_rejected(size):
    with pytest.raises(ValueError, match="size"):
        run(api_key="sk-test", model="gpt-image-2.5-flare", prompt="x", size=size)


def test_transparent_jpeg_rejected():
    with pytest.raises(ValueError, match="jpeg"):
        run(api_key="sk-test", model="gpt-image-2.5-flare", prompt="x",
            background="transparent", output_format="jpeg")


def test_edit_with_references():
    ref = torch.rand(1, 16, 16, 3)
    _, _, client, _ = run(
        api_key="sk-test", model="gpt-image-2.5-sunburst", prompt="combine", quality="max",
        reference_image_1=ref, reference_image_3=ref,
    )
    client.images.generate.assert_not_called()
    kwargs = client.images.edit.call_args.kwargs
    assert kwargs["model"] == "gpt-image-2.5-sunburst"
    assert kwargs["quality"] == "max"
    assert [f[0] for f in kwargs["image"]] == ["reference_1_0.png", "reference_3_0.png"]
    assert Image.open(io.BytesIO(kwargs["image"][0][1])).size == (16, 16)


def test_env_key_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    _, _, _, ctor = run(api_key="  ", model="gpt-image-2.5-flare", prompt="x")
    ctor.assert_called_once_with(api_key="sk-env")


def test_input_types():
    types = nodes.OpenAIImageGeneration.INPUT_TYPES()
    assert types["required"]["model"][0] == ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]
    assert types["required"]["quality"][0] == ["auto", "low", "medium", "high", "xhigh", "max"]
    assert types["required"]["background"][0] == ["auto", "opaque", "transparent"]
    assert types["required"]["output_format"][0] == ["png", "jpeg", "webp"]
    assert sorted(types["optional"]) == [f"reference_image_{i}" for i in range(1, 6)]
