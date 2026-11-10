from diffusers import DiffusionPipeline
from tools.prompt_utils import rewrite
from model_paths import QWEN_IMAGE_MODEL
import torch

# Initialize the pipeline from local weights
pipe = DiffusionPipeline.from_pretrained(
    QWEN_IMAGE_MODEL,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe = pipe.to("cuda")

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

prompt = "一只可爱的小猫坐在花园里"  # Chinese prompt
prompt = rewrite(prompt)

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example.png")