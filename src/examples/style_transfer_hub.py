"""
🎨 Style Transfer Hub - New Feature for Qwen-Image
One-click style transfer between different artistic styles

This feature provides a comprehensive style transfer system that allows users to
transform images between various artistic styles including impressionism, cubism,
anime, watercolor, sketch, and many more.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import json
import os


@dataclass
class StylePreset:
    """Represents a predefined artistic style."""
    name: str
    description: str
    prompt_template: str
    strength: float = 0.8
    negative_prompt: str = ""
    guidance_scale: float = 7.5


class StyleTransferHub:
    """
    Comprehensive style transfer system for Qwen-Image.
    Provides one-click style transformation with presets and customization.
    """

    def __init__(self, model_path: str = "Qwen/Qwen-Image"):
        self.model_path = model_path
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.style_presets = self._load_style_presets()

    def _load_style_presets(self) -> Dict[str, StylePreset]:
        """Load predefined artistic style presets."""
        return {
            "impressionism": StylePreset(
                name="Impressionism",
                description="Soft brushstrokes, light effects, outdoor scenes like Monet",
                prompt_template="in the style of impressionism, soft brushstrokes, light and shadow play, outdoor scene, {original_description}, impressionist painting, monet style",
                strength=0.85,
                guidance_scale=8.0
            ),

            "cubism": StylePreset(
                name="Cubism",
                description="Geometric shapes, fragmented forms, Picasso-inspired",
                prompt_template="in the style of cubism, geometric shapes, fragmented forms, multiple perspectives, {original_description}, picasso inspired, analytical cubism",
                strength=0.9,
                guidance_scale=9.0
            ),

            "anime": StylePreset(
                name="Anime",
                description="Japanese animation style with vibrant colors and expressive characters",
                prompt_template="anime style, vibrant colors, expressive eyes, detailed backgrounds, {original_description}, studio ghibli inspired, high quality anime art",
                strength=0.8,
                guidance_scale=7.0
            ),

            "watercolor": StylePreset(
                name="Watercolor",
                description="Soft watercolor painting with flowing colors and textures",
                prompt_template="watercolor painting, soft flowing colors, paper texture, wet paint effects, {original_description}, artistic watercolor, delicate brushstrokes",
                strength=0.75,
                guidance_scale=6.5
            ),

            "sketch": StylePreset(
                name="Pencil Sketch",
                description="Black and white pencil sketch with fine line work",
                prompt_template="pencil sketch, black and white, fine line work, detailed shading, {original_description}, charcoal drawing, artistic sketch",
                strength=0.9,
                guidance_scale=8.5,
                negative_prompt="color, colorful, bright colors"
            ),

            "oil_painting": StylePreset(
                name="Oil Painting",
                description="Rich oil painting with thick brushstrokes and texture",
                prompt_template="oil painting, thick brushstrokes, rich colors, canvas texture, {original_description}, classical oil painting, van gogh style",
                strength=0.8,
                guidance_scale=7.5
            ),

            "pop_art": StylePreset(
                name="Pop Art",
                description="Bold colors, comic book style, Andy Warhol inspired",
                prompt_template="pop art style, bold colors, comic book, halftone dots, {original_description}, andy warhol inspired, vibrant pop culture",
                strength=0.85,
                guidance_scale=8.0
            ),

            "surrealism": StylePreset(
                name="Surrealism",
                description="Dream-like scenes, Dali-inspired melting clocks and impossible objects",
                prompt_template="surrealism, dream-like, melting objects, impossible architecture, {original_description}, dali inspired, surreal art",
                strength=0.9,
                guidance_scale=9.0
            ),

            "minimalist": StylePreset(
                name="Minimalist",
                description="Clean, simple, geometric shapes with limited color palette",
                prompt_template="minimalist art, clean lines, geometric shapes, limited colors, {original_description}, modern minimalist, simple composition",
                strength=0.7,
                guidance_scale=6.0
            ),

            "vintage_photo": StylePreset(
                name="Vintage Photography",
                description="Retro film photography with grain and aged effects",
                prompt_template="vintage photograph, film grain, aged photo, sepia tones, {original_description}, retro photography, old film look",
                strength=0.8,
                guidance_scale=7.0
            )
        }

    def load_pipeline(self):
        """Load the Qwen-Image pipeline for style transfer."""
        try:
            from diffusers import DiffusionPipeline

            print("[INFO] Loading Style Transfer Hub pipeline...")

            self.pipeline = DiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            if self.device == "cuda":
                self.pipeline.to(self.device)

            print("[OK] Style Transfer Hub ready!")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
            return False

    def analyze_image_content(self, image: Image.Image) -> str:
        """
        Analyze image content to generate descriptive prompts.
        This is a simplified version - in production, you'd use CLIP or similar.
        """
        # Basic content analysis (simplified)
        width, height = image.size

        # Determine aspect and basic composition
        if width > height:
            aspect = "landscape"
        elif height > width:
            aspect = "portrait"
        else:
            aspect = "square"

        # Generate basic description
        description = f"{aspect} composition image"

        # Add color information
        img_array = np.array(image.convert('RGB'))
        colors = np.mean(img_array, axis=(0, 1))
        brightness = np.mean(colors)

        if brightness > 180:
            description += ", bright and light"
        elif brightness < 80:
            description += ", dark and moody"
        else:
            description += ", balanced lighting"

        return description

    def transfer_style(
        self,
        image: Image.Image,
        style_name: str,
        custom_prompt: Optional[str] = None,
        strength: Optional[float] = None,
        **kwargs
    ) -> Tuple[Image.Image, Dict]:
        """
        Transfer image to specified artistic style.

        Args:
            image: Input PIL image
            style_name: Name of style preset or 'custom'
            custom_prompt: Custom prompt for style transfer
            strength: Style transfer strength (0-1)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (styled_image, metadata_dict)
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")

        if style_name not in self.style_presets and style_name != "custom":
            available_styles = list(self.style_presets.keys())
            raise ValueError(f"Unknown style '{style_name}'. Available: {available_styles}")

        # Get style configuration
        if style_name == "custom":
            if not custom_prompt:
                raise ValueError("custom_prompt required for custom style")
            style_config = StylePreset(
                name="Custom",
                description="User-defined style",
                prompt_template=custom_prompt,
                strength=strength or 0.8
            )
        else:
            style_config = self.style_presets[style_name]
            if strength is not None:
                style_config.strength = strength

        # Analyze original image content
        original_description = self.analyze_image_content(image)

        # Generate styled prompt
        styled_prompt = style_config.prompt_template.format(
            original_description=original_description
        )

        # Prepare generation parameters
        gen_params = {
            "prompt": styled_prompt,
            "image": image,
            "strength": style_config.strength,
            "guidance_scale": style_config.guidance_scale,
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
            "generator": kwargs.get("generator", torch.Generator(device=self.device).manual_seed(42)),
        }

        # Add negative prompt if specified
        if style_config.negative_prompt:
            gen_params["negative_prompt"] = style_config.negative_prompt

        # Override with user parameters
        gen_params.update(kwargs)

        print(f"[INFO] Applying {style_config.name} style transfer...")
        print(f"[INFO] Prompt: {styled_prompt[:100]}...")

        # Generate styled image
        with torch.inference_mode():
            result = self.pipeline(**gen_params)

        styled_image = result.images[0] if hasattr(result, 'images') else result

        # Prepare metadata
        metadata = {
            "style_name": style_config.name,
            "style_description": style_config.description,
            "original_description": original_description,
            "prompt_used": styled_prompt,
            "strength": style_config.strength,
            "guidance_scale": style_config.guidance_scale,
            "generation_params": gen_params
        }

        return styled_image, metadata

    def batch_style_transfer(
        self,
        images: List[Image.Image],
        style_name: str,
        **kwargs
    ) -> List[Tuple[Image.Image, Dict]]:
        """
        Apply style transfer to multiple images.

        Args:
            images: List of PIL images
            style_name: Style to apply
            **kwargs: Additional parameters

        Returns:
            List of (styled_image, metadata) tuples
        """
        results = []
        for i, image in enumerate(images):
            print(f"[INFO] Processing image {i+1}/{len(images)}")
            try:
                styled_image, metadata = self.transfer_style(image, style_name, **kwargs)
                results.append((styled_image, metadata))
            except Exception as e:
                print(f"[ERROR] Failed to process image {i+1}: {e}")
                results.append((None, {"error": str(e)}))

        return results

    def create_custom_style(
        self,
        name: str,
        description: str,
        prompt_template: str,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        negative_prompt: str = ""
    ) -> StylePreset:
        """
        Create a custom style preset.

        Args:
            name: Style name
            description: Style description
            prompt_template: Prompt template with {original_description} placeholder
            strength: Style strength
            guidance_scale: Guidance scale
            negative_prompt: Negative prompt

        Returns:
            StylePreset object
        """
        custom_style = StylePreset(
            name=name,
            description=description,
            prompt_template=prompt_template,
            strength=strength,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt
        )

        # Add to presets
        self.style_presets[name.lower().replace(" ", "_")] = custom_style

        return custom_style

    def list_available_styles(self) -> Dict[str, str]:
        """List all available style presets."""
        return {name: preset.description for name, preset in self.style_presets.items()}

    def save_style_preset(self, style_name: str, filepath: str):
        """Save a style preset to JSON file."""
        if style_name not in self.style_presets:
            raise ValueError(f"Style '{style_name}' not found")

        preset = self.style_presets[style_name]
        preset_dict = {
            "name": preset.name,
            "description": preset.description,
            "prompt_template": preset.prompt_template,
            "strength": preset.strength,
            "guidance_scale": preset.guidance_scale,
            "negative_prompt": preset.negative_prompt
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(preset_dict, f, indent=2, ensure_ascii=False)

    def load_style_preset(self, filepath: str) -> str:
        """Load a style preset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            preset_dict = json.load(f)

        style_name = preset_dict["name"].lower().replace(" ", "_")

        self.style_presets[style_name] = StylePreset(**preset_dict)

        return style_name


def demonstrate_style_transfer_hub():
    """
    Demonstrate the Style Transfer Hub functionality.
    """
    print("=" * 60)
    print("QWEN-IMAGE STYLE TRANSFER HUB")
    print("=" * 60)
    print()

    print("NEW FEATURE: One-click artistic style transfer")
    print()

    # Initialize hub
    hub = StyleTransferHub()

    print("AVAILABLE STYLES:")
    styles = hub.list_available_styles()
    for name, desc in styles.items():
        print(f"  * {name}: {desc}")
    print()

    print("USAGE EXAMPLES:")
    print("""
# Basic style transfer
from style_transfer_hub import StyleTransferHub

hub = StyleTransferHub()
hub.load_pipeline()

# Load your image
image = Image.open("your_photo.jpg")

# Apply impressionist style
styled_image, metadata = hub.transfer_style(image, "impressionism")

# Save result
styled_image.save("impressionist_result.jpg")

# Batch processing
images = [Image.open(f"img{i}.jpg") for i in range(5)]
results = hub.batch_style_transfer(images, "anime")

# Create custom style
custom_style = hub.create_custom_style(
    name="Cyberpunk",
    description="Neon-lit futuristic cityscape",
    prompt_template="cyberpunk style, neon lights, futuristic city, {original_description}, high tech, digital art"
)

# Apply custom style
styled_image, metadata = hub.transfer_style(image, "cyberpunk")
""")

    print("KEY FEATURES:")
    print("* 10+ built-in artistic styles")
    print("* Custom style creation")
    print("* Batch processing support")
    print("* Automatic content analysis")
    print("* Style strength control")
    print("* Metadata preservation")
    print("* Preset saving/loading")
    print()

    print("PERFECT FOR:")
    print("* Artists exploring different styles")
    print("* Content creators")
    print("* Social media graphics")
    print("* Educational materials")
    print("* Creative experimentation")
    print()


if __name__ == "__main__":
    demonstrate_style_transfer_hub()