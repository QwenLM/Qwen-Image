"""Local model weight paths. Override via env vars if needed."""
import os

# Default layout on server:
# /data/models/Qwen-Image-2512
# /data/models/Qwen-Image-Edit-2511
# /data/img-code/Qwen-Image  (this repo)
MODELS_ROOT = os.environ.get("QWEN_MODELS_ROOT", "/data/models")

QWEN_IMAGE_MODEL = os.environ.get(
    "QWEN_IMAGE_MODEL",
    os.path.join(MODELS_ROOT, "Qwen-Image-2512"),
)

QWEN_IMAGE_EDIT_MODEL = os.environ.get(
    "QWEN_IMAGE_EDIT_MODEL",
    os.path.join(MODELS_ROOT, "Qwen-Image-Edit-2511"),
)
