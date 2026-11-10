"""原生 Qwen 协议单测：不需要 GPU 或 diffusers。"""
import pytest
from pydantic import TypeAdapter, ValidationError

from protocol import EditRequest, GenerateRequest, ImageRequest

REQUEST = TypeAdapter(ImageRequest)


def parse(payload):
    return REQUEST.validate_python(payload)


def test_generate_uses_native_pipeline_fields():
    req = parse(
        {
            "model": "qwen-image",
            "prompt": "a cat",
            "width": 1664,
            "height": 928,
            "true_cfg_scale": 4.0,
            "guidance_scale": 1.0,
            "num_inference_steps": 50,
            "num_images_per_prompt": 2,
        }
    )
    assert isinstance(req, GenerateRequest)
    task = req.to_task()
    assert task.op == "t2i"
    assert (task.width, task.height) == (1664, 928)
    assert task.num_images_per_prompt == 2
    assert task.num_inference_steps == 50


def test_edit_requires_images():
    with pytest.raises(ValidationError):
        parse({"model": "qwen-image-edit", "prompt": "make it blue"})


def test_edit_uses_images_field():
    req = parse(
        {
            "model": "qwen-image-edit",
            "prompt": "make it blue",
            "images": ["aGk=", "aGk="],
        }
    )
    assert isinstance(req, EditRequest)
    task = req.to_task()
    assert task.op == "edit"
    assert task.images_b64 == ["aGk=", "aGk="]
    assert task.width is None
    assert task.height is None


def test_generate_rejects_edit_only_field():
    with pytest.raises(ValidationError):
        parse(
            {
                "model": "qwen-image",
                "prompt": "a cat",
                "images": ["AAAA"],
            }
        )


def test_openai_compatibility_fields_are_rejected():
    with pytest.raises(ValidationError):
        parse(
            {
                "model": "qwen-image",
                "prompt": "a cat",
                "size": "1664x928",
            }
        )
    with pytest.raises(ValidationError):
        parse(
            {
                "model": "qwen-image",
                "prompt": "a cat",
                "response_format": "url",
            }
        )


def test_seed_and_native_parameter_bounds():
    req = parse({"model": "qwen-image", "prompt": "x"})
    assert req.seed == 42
    with pytest.raises(ValidationError):
        parse({"model": "qwen-image", "prompt": "x", "seed": -1})
    with pytest.raises(ValidationError):
        parse({"model": "qwen-image", "prompt": "x", "num_inference_steps": 0})