"""DashScope compatibility adapter tests; no GPU or network required."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from errors import InvalidRequest
from qwen_image_dashscope import (
    DashScopeRequest,
    from_job,
    to_task,
)


T2I_REQUEST = {
    "model": "qwen-image-plus",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [{"text": "a cat in a spacesuit"}],
            }
        ]
    },
    "parameters": {
        "size": "1664*928",
        "n": 2,
        "seed": 7,
        "num_inference_steps": 40,
    },
}


def test_t2i_request_maps_to_local_task():
    task = to_task(DashScopeRequest.model_validate(T2I_REQUEST))

    assert task.op == "t2i"
    assert task.prompt == "a cat in a spacesuit"
    assert (task.width, task.height) == (1664, 928)
    assert task.num_images_per_prompt == 2
    assert task.seed == 7
    assert task.num_inference_steps == 40


def test_edit_request_accepts_data_url_images():
    request = DashScopeRequest.model_validate(
        {
            "model": "qwen-image-edit-plus",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": "data:image/png;base64,aGk="},
                            {"text": "make it blue"},
                        ],
                    }
                ]
            },
        }
    )

    task = to_task(request)

    assert task.op == "edit"
    assert task.prompt == "make it blue"
    assert task.images_b64 == ["aGk="]
    assert task.width is None
    assert task.height is None


def test_t2i_rejects_image_content():
    request = DashScopeRequest.model_validate(
        {
            "model": "qwen-image-plus",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": "data:image/png;base64,aGk="},
                            {"text": "describe"},
                        ],
                    }
                ]
            },
        }
    )

    with pytest.raises(InvalidRequest, match="cannot contain image"):
        to_task(request)


def test_invalid_size_and_model_are_rejected():
    with pytest.raises(InvalidRequest, match="size"):
        to_task(
            DashScopeRequest.model_validate(
                {
                    **T2I_REQUEST,
                    "parameters": {"size": "1920x1080"},
                }
            )
        )

    with pytest.raises(InvalidRequest, match="unsupported"):
        to_task(
            DashScopeRequest.model_validate(
                {**T2I_REQUEST, "model": "qwen-unknown"}
            )
        )


def test_response_uses_public_local_file_urls():
    response = from_job(
        SimpleNamespace(
            job_id="job-1",
            blobs=["a" * 64, "b" * 64],
        ),
        public_base_url="https://images.example.test/",
    )

    assert response.request_id == "job-1"
    assert response.usage.image_count == 2
    assert [
        item.image
        for item in response.output.choices[0].message.content
    ] == [
        f"https://images.example.test/files/{'a' * 64}",
        f"https://images.example.test/files/{'b' * 64}",
    ]


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        DashScopeRequest.model_validate({**T2I_REQUEST, "extra": True})