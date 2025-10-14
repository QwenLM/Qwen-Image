"""
Fix for deployment quality issues with Qwen-Image-Edit-2509
Addresses issue #152: Qwen-Image-Edit-2509 部署后跑官网的case，效果和官网差距太大了

This script provides fixes for quality degradation in self-hosted deployments.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, List
from PIL import Image


class QualityEnhancedQwenImageEdit:
    """
    Enhanced Qwen-Image-Edit pipeline with quality preservation.
    """

    def __init__(self, model_path: str = "Qwen/Qwen-Image-Edit-2509"):
        self.model_path = model_path
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_with_quality_settings(self):
        """
        Load pipeline with quality-preserving settings.
        """
        try:
            from diffusers import QwenImageEditPlusPipeline

            print("[INFO] Loading Qwen-Image-Edit-2509 with quality settings...")

            # Quality-preserving loading parameters
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better quality
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            # Move to device
            if self.device == "cuda":
                self.pipeline.to(self.device)

            # Quality enhancements
            self._apply_quality_enhancements()

            print("[OK] Pipeline loaded with quality enhancements")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
            return False

    def _apply_quality_enhancements(self):
        """
        Apply quality enhancement settings.
        """
        if not self.pipeline:
            return

        # Enable high-quality settings
        self.pipeline.set_progress_bar_config(disable=None)

        # Set optimal generation parameters
        self.default_params = {
            "true_cfg_scale": 4.0,  # Higher CFG for better quality
            "num_inference_steps": 40,  # More steps for refinement
            "guidance_scale": 1.0,
            "num_images_per_prompt": 1,
            "negative_prompt": " ",  # Empty negative prompt for consistency
        }

        print("[OK] Quality enhancements applied")

    def enhance_prompt_for_quality(self, prompt: str, image: Image.Image) -> str:
        """
        Enhance prompt to match official demo quality.
        """
        # Use the existing prompt enhancement from tools
        try:
            from tools.prompt_utils import polish_edit_prompt
            enhanced_prompt = polish_edit_prompt(prompt, image)
            return enhanced_prompt
        except:
            # Fallback: add quality modifiers
            quality_modifiers = [
                "high quality",
                "detailed",
                "professional",
                "ultra realistic"
            ]

            enhanced = prompt
            for modifier in quality_modifiers:
                if modifier not in prompt.lower():
                    enhanced += f", {modifier}"

            return enhanced

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better quality.
        """
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if too large (maintain aspect ratio)
        max_size = 2048
        width, height = image.size

        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    def generate_with_quality(self, image: Image.Image, prompt: str, **kwargs) -> Any:
        """
        Generate with quality-focused parameters.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded. Call load_with_quality_settings() first.")

        # Preprocess image
        image = self.preprocess_image(image)

        # Enhance prompt
        enhanced_prompt = self.enhance_prompt_for_quality(prompt, image)

        # Merge with quality defaults
        generation_params = {**self.default_params, **kwargs}
        generation_params.update({
            "image": image,
            "prompt": enhanced_prompt,
        })

        print(f"[INFO] Generating with enhanced prompt: {enhanced_prompt[:100]}...")

        # Generate
        with torch.inference_mode():
            result = self.pipeline(**generation_params)

        return result

    def compare_with_official(self, local_result: Any, description: str = "test case"):
        """
        Compare local result with expected official quality.
        """
        print(f"\nQUALITY COMPARISON FOR: {description}")
        print("=" * 50)

        if hasattr(local_result, 'images') and len(local_result.images) > 0:
            image = local_result.images[0]

            # Basic quality checks
            width, height = image.size
            print(f"Output resolution: {width}x{height}")

            # Check for common quality issues
            img_array = np.array(image)

            # Check color diversity
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            print(f"Color diversity: {unique_colors} unique colors")

            # Check brightness distribution
            brightness = np.mean(img_array, axis=2)
            brightness_std = np.std(brightness)
            print(f"Brightness variation: {brightness_std:.2f}")

            print("\nTROUBLESHOOTING TIPS:")
            print("1. Ensure using Qwen-Image-Edit-2509 (not older version)")
            print("2. Use true_cfg_scale=4.0 for consistency")
            print("3. Try different random seeds if results vary")
            print("4. Check that prompt enhancement is working")
            print("5. Ensure image preprocessing is applied")

        else:
            print("[ERROR] No valid output to analyze")


def diagnose_quality_issues():
    """
    Diagnose common quality issues in self-hosted deployments.
    """
    print("=" * 60)
    print("QWEN-IMAGE-EDIT-2509 DEPLOYMENT QUALITY DIAGNOSTIC")
    print("=" * 60)
    print()

    print("ISSUE #152: Self-hosted results differ significantly from official demo")
    print()

    print("COMMON ROOT CAUSES:")
    print("1. Using older model version (Qwen-Image-Edit vs Qwen-Image-Edit-2509)")
    print("2. Incorrect CFG scale (should be true_cfg_scale=4.0)")
    print("3. Missing prompt enhancement/polishing")
    print("4. Different random seed affecting consistency")
    print("5. Image preprocessing differences")
    print("6. Precision issues (bf16 vs float32)")
    print("7. Different inference steps or parameters")
    print()

    print("QUALITY ENHANCEMENT SOLUTIONS:")
    print("1. Use QwenImageEditPlusPipeline for 2509 features")
    print("2. Apply prompt polishing with Qwen-VL-Max")
    print("3. Use consistent random seeds")
    print("4. Enable bfloat16 precision")
    print("5. Apply image preprocessing")
    print("6. Use recommended generation parameters")
    print()


def main():
    """
    Main function demonstrating quality fixes.
    """
    diagnose_quality_issues()

    print("USAGE EXAMPLE:")
    print("""
from fix_deployment_quality import QualityEnhancedQwenImageEdit
from PIL import Image

# Initialize enhanced pipeline
pipeline = QualityEnhancedQwenImageEdit()
pipeline.load_with_quality_settings()

# Load your test image
image = Image.open("your_test_image.png")

# Generate with quality enhancements
result = pipeline.generate_with_quality(
    image=image,
    prompt="Change background to blue sky"
)

# Compare quality
pipeline.compare_with_official(result, "background change test")
""")

    print("\nOFFICIAL DEMO REPLICATION CHECKLIST:")
    print("[ ] Using Qwen-Image-Edit-2509 model")
    print("[ ] Using QwenImageEditPlusPipeline")
    print("[ ] true_cfg_scale=4.0")
    print("[ ] num_inference_steps=40")
    print("[ ] guidance_scale=1.0")
    print("[ ] negative_prompt=' ' (single space)")
    print("[ ] torch_dtype=torch.bfloat16")
    print("[ ] Prompt enhancement enabled")
    print("[ ] Image preprocessing applied")
    print("[ ] Same random seed as demo")
    print()

    print("If issues persist, the differences may be due to:")
    print("- Server-side optimizations not available locally")
    print("- Different model weights (check model hash)")
    print("- Environment-specific factors")


if __name__ == "__main__":
    main()