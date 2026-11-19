import base64
import io
import sys
from pathlib import Path
from unittest import mock

from types import SimpleNamespace

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


KEYS = {"provider": "auto", "openrouter_api_key": "", "openai_api_key": ""}


def run(alpha=255, ok=True, **kwargs):
    """Run the node with requests.post (OpenRouter) and openai.OpenAI both mocked."""
    kwargs = {**KEYS, **kwargs}
    response = mock.MagicMock(ok=ok, status_code=200 if ok else 401, text="err")
    response.json.return_value = {"data": [{"b64_json": fake_b64(alpha=alpha), "media_type": "image/png"}]}
    client = mock.MagicMock()
    result = SimpleNamespace(data=[SimpleNamespace(b64_json=fake_b64(alpha=alpha))])
    client.images.generate.return_value = result
    client.images.edit.return_value = result
    with mock.patch("requests.post", return_value=response) as post, \
            mock.patch("openai.OpenAI", return_value=client) as ctor:
        out, mask = nodes.OpenAIImageGeneration().generate(**kwargs)
    return out, mask, post, client, ctor


def test_openrouter_generate():
    out, mask, post, client, ctor = run(openrouter_api_key="sk-or", model="gpt-image-2.5-flare", prompt="a cat")
    assert post.call_args.args[0] == "https://openrouter.ai/api/v1/images"
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-or"
    assert post.call_args.kwargs["json"] == {
        "model": "openai/gpt-image-2.5-flare", "prompt": "a cat", **DEFAULTS
    }
    ctor.assert_not_called()
    assert out.shape == (1, 4, 8, 3)
    assert torch.allclose(out[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert mask.shape == (1, 4, 8) and torch.all(mask == 0)


def test_openrouter_options_forwarded():
    _, mask, post, _, _ = run(
        alpha=0, openrouter_api_key="sk-or", model="gpt-image-2.5-flare", prompt="logo",
        size="1536X1024", quality="high", background="transparent", output_format="webp",
    )
    assert post.call_args.kwargs["json"] == {
        "model": "openai/gpt-image-2.5-flare", "prompt": "logo", "size": "1536x1024",
        "quality": "high", "background": "transparent", "output_format": "webp",
    }
    assert torch.all(mask == 1)


def test_openrouter_references():
    ref = torch.rand(1, 16, 16, 3)
    _, _, post, _, _ = run(
        openrouter_api_key="sk-or", model="gpt-image-2.5-sunburst", prompt="combine", quality="max",
        reference_image_1=ref, reference_image_3=ref,
    )
    payload = post.call_args.kwargs["json"]
    assert payload["model"] == "openai/gpt-image-2.5-sunburst"
    refs = payload["input_references"]
    assert len(refs) == 2 and refs[0]["type"] == "image_url"
    url = refs[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size == (16, 16)


def test_openrouter_http_error():
    with pytest.raises(RuntimeError, match="401"):
        run(ok=False, openrouter_api_key="sk-or", model="gpt-image-2.5-flare", prompt="x")


def test_openai_generate():
    out, mask, post, client, ctor = run(openai_api_key="sk-oa", model="gpt-image-2.5-flare", prompt="a cat")
    ctor.assert_called_once_with(api_key="sk-oa")
    client.images.generate.assert_called_once_with(
        model="gpt-image-2.5-flare", prompt="a cat", **DEFAULTS
    )
    client.images.edit.assert_not_called()
    post.assert_not_called()
    assert out.shape == (1, 4, 8, 3)


def test_openai_edit_with_references():
    ref = torch.rand(1, 16, 16, 3)
    _, _, _, client, _ = run(
        openai_api_key="sk-oa", model="gpt-image-2.5-sunburst", prompt="combine",
        reference_image_1=ref, reference_image_3=ref,
    )
    client.images.generate.assert_not_called()
    kwargs = client.images.edit.call_args.kwargs
    assert [f[0] for f in kwargs["image"]] == ["reference_1_0.png", "reference_2_0.png"]


def test_auto_prefers_openrouter_when_both_keys():
    _, _, post, _, ctor = run(openrouter_api_key="sk-or", openai_api_key="sk-oa",
                              model="gpt-image-2.5-flare", prompt="x")
    post.assert_called_once()
    ctor.assert_not_called()


def test_forced_provider_overrides_auto():
    _, _, post, _, ctor = run(provider="openai", openrouter_api_key="sk-or", openai_api_key="sk-oa",
                              model="gpt-image-2.5-flare", prompt="x")
    ctor.assert_called_once_with(api_key="sk-oa")
    post.assert_not_called()


def test_env_key_fallbacks(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-or")
    _, _, post, _, _ = run(model="gpt-image-2.5-flare", prompt="x")
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer sk-env-or"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-oa")
    _, _, _, _, ctor = run(model="gpt-image-2.5-flare", prompt="x")
    ctor.assert_called_once_with(api_key="sk-env-oa")


def test_no_key_errors(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        run(model="gpt-image-2.5-flare", prompt="x")
    with pytest.raises(ValueError, match="openai_api_key"):
        run(provider="openai", openrouter_api_key="sk-or", model="gpt-image-2.5-flare", prompt="x")


@pytest.mark.parametrize("size", ["1024", "1000x1024", "0x1024", "big"])
def test_invalid_size_rejected(size):
    with pytest.raises(ValueError, match="size"):
        run(openrouter_api_key="sk-or", model="gpt-image-2.5-flare", prompt="x", size=size)


def test_transparent_jpeg_rejected():
    with pytest.raises(ValueError, match="jpeg"):
        run(openrouter_api_key="sk-or", model="gpt-image-2.5-flare", prompt="x",
            background="transparent", output_format="jpeg")


def test_input_types():
    types = nodes.OpenAIImageGeneration.INPUT_TYPES()
    assert types["required"]["model"][0] == ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"]
    assert types["required"]["quality"][0] == ["auto", "low", "medium", "high", "xhigh", "max"]
    assert types["required"]["background"][0] == ["auto", "opaque", "transparent"]
    assert types["required"]["output_format"][0] == ["png", "jpeg", "webp"]
    assert sorted(types["optional"]) == [f"reference_image_{i}" for i in range(1, 6)]
