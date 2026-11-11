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
    "model": "Qwen-Image-2512",
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
            "model": "Qwen-Image-Edit-2511",
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
            "model": "Qwen-Image-2512",
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
                    "parameters": {"size": "1920-1080"},
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


Z_IMAGE_REQUEST = {
    "model": "Z-Image-Turbo",
    "input": {
        "messages": [
            {
                "role": "user",
                "content": [{"text": "一只坐着的橘黄色的猫"}],
            }
        ]
    },
    "parameters": {
        "size": "1024*1536",
        "prompt_extend": False,
        "seed": 42,
    },
}


def test_z_image_turbo_maps_to_local_task_with_official_defaults():
    task = to_task(DashScopeRequest.model_validate(Z_IMAGE_REQUEST))

    assert task.op == "z_image_turbo"
    assert task.prompt == "一只坐着的橘黄色的猫"
    assert (task.width, task.height) == (1024, 1536)
    assert task.num_images_per_prompt == 1
    assert task.seed == 42
    assert task.num_inference_steps == 9
    assert task.guidance_scale == 0.0


def test_z_image_maps_to_base_worker_defaults():
    task = to_task(
        DashScopeRequest.model_validate(
            {
                **Z_IMAGE_REQUEST,
                "model": "Z-Image",
                "parameters": {"size": "1280*720", "seed": 7},
            }
        )
    )

    assert task.op == "z_image"
    assert (task.width, task.height) == (1280, 720)
    assert task.num_inference_steps == 50
    assert task.guidance_scale == 4.0


def test_z_image_default_size_is_official():
    task = to_task(
        DashScopeRequest.model_validate(
            {
                "model": "Z-Image-Turbo",
                "input": Z_IMAGE_REQUEST["input"],
                "parameters": {"seed": 1},
            }
        )
    )

    assert (task.width, task.height) == (1024, 1536)


def test_z_image_rejects_n_not_one_and_images():
    with pytest.raises(InvalidRequest, match="n"):
        to_task(
            DashScopeRequest.model_validate(
                {
                    **Z_IMAGE_REQUEST,
                    "parameters": {"size": "1024*1024", "n": 2, "seed": 1},
                }
            )
        )

    with pytest.raises(InvalidRequest, match="cannot contain image"):
        to_task(
            DashScopeRequest.model_validate(
                {
                    "model": "Z-Image",
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"image": "data:image/png;base64,aGk="},
                                    {"text": "cat"},
                                ],
                            }
                        ]
                    },
                }
            )
        )


def test_z_image_rejects_invalid_pixel_area_and_prompt_extend():
    with pytest.raises(InvalidRequest, match="size"):
        to_task(
            DashScopeRequest.model_validate(
                {
                    **Z_IMAGE_REQUEST,
                    "parameters": {"size": "256*256", "seed": 1},
                }
            )
        )

    with pytest.raises(InvalidRequest, match="prompt_extend"):
        to_task(
            DashScopeRequest.model_validate(
                {
                    **Z_IMAGE_REQUEST,
                    "parameters": {
                        "size": "1024*1024",
                        "prompt_extend": True,
                        "seed": 1,
                    },
                }
            )
        )
