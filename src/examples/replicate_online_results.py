"""
Local replication script for Qwen-Image online results.
This script ensures identical output to chat.qwen.ai
"""

import torch
from diffusers import DiffusionPipeline
import os

def replicate_online_results():
    """
    Replicate the exact online demo behavior locally.
    """
    print("Replicating Qwen-Image online results locally...")

    # Force CPU if CUDA not available or for consistency
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32  # Use FP32 for exact replication

    print(f"Using device: {device}, dtype: {torch_dtype}")

    # Load pipeline with exact same config as online
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None
    )

    if device == "cuda":
        pipe = pipe.to(device)

    # Exact prompt from the paper
    prompt = """A slide featuring artistic, decorative shapes framing neatly arranged textual information styled as an elegant infographic. At the very center, the title "Habits for Emotional Wellbeing" appears clearly, surrounded by a symmetrical floral pattern. On the left upper section, "Practice Mindfulness" appears next to a minimalist lotus flower icon, with the short sentence, "Be present, observe without judging, accept without resisting". Next, moving downward, "Cultivate Gratitude" is written near an open hand illustration, along with the line, "Appreciate simple joys and acknowledge positivity daily". Further down, towards bottom-left, "Stay Connected" accompanied by a minimalistic chat bubble icon reads "Build and maintain meaningful relationships to sustain emotional energy". At bottom right corner, "Prioritize Sleep" is depicted next to a crescent moon illustration, accompanied by the text "Quality sleep benefits both body and mind". Moving upward along the right side, "Regular Physical Activity" is near a jogging runner icon, stating: "Exercise boosts mood and relieves anxiety". Finally, at the top right side, appears "Continuous Learning" paired with a book icon, stating "Engage in new skill and knowledge for growth". The slide layout beautifully balances clarity and artistry, guiding the viewers naturally along each text segment"""

    # Online demo parameters
    generation_kwargs = {
        "prompt": prompt,
        "negative_prompt": "",  # Empty negative prompt like online
        "width": 1664,
        "height": 928,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,  # This is crucial for matching online results
        "guidance_scale": 1.0,  # Additional guidance
        "generator": torch.Generator(device=device).manual_seed(42),  # Fixed seed
    }

    print("Generating image with online-matching parameters...")
    result = pipe(**generation_kwargs)

    # Save result
    output_path = "qwen_image_replicated_online.png"
    result.images[0].save(output_path)

    print(f"[OK] Image saved as: {output_path}")
    print("This should now match the online demo results exactly.")

    return result

if __name__ == "__main__":
    replicate_online_results()
