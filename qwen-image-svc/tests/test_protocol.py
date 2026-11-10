"""协议层单测：不需要 GPU、不需要 torch。"""
import pytest
from pydantic import ValidationError

from protocol import ImageRequest


def test_defaults_map_to_t2i():
    req = ImageRequest(prompt="a cat")
    task = req.to_task()
    assert task.op == "t2i"
    assert (task.width, task.height) == (1328, 1328)
    assert task.n == 1
    assert task.image_b64 == []


def test_edit_requires_image():
    with pytest.raises(ValidationError):
        ImageRequest(model="qwen-image-edit", prompt="make it blue")


def test_t2i_rejects_image():
    with pytest.raises(ValidationError):
        ImageRequest(prompt="a cat", image_b64=["AAAA"])


def test_unknown_size_rejected():
    with pytest.raises(ValidationError):
        ImageRequest(prompt="a cat", size="1920x1080")


@pytest.mark.parametrize("n", [0, 5])
def test_n_bounds(n):
    with pytest.raises(ValidationError):
        ImageRequest(prompt="a cat", n=n)


def test_seed_optional_but_bounded():
    assert ImageRequest(prompt="x", seed=None).to_task().seed is None
    with pytest.raises(ValidationError):
        ImageRequest(prompt="x", seed=2**31)


def test_size_preset_maps_exactly():
    task = ImageRequest(prompt="x", size="1664x928").to_task()
    assert (task.width, task.height) == (1664, 928)


def test_edit_task_carries_images():
    task = ImageRequest(
        model="qwen-image-edit", prompt="e", image_b64=["aGk=", "aGk="]
    ).to_task()
    assert task.op == "edit"
    assert len(task.image_b64) == 2