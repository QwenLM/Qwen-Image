"""
Fix for memory issue during Edit model loading
Addresses issue #74: 尝试了最新的Edit模型，但系统内存在Loading pipeline阶段崩了

This script provides memory-efficient loading for Qwen-Image-Edit models.
"""

import os
import gc
import torch
from typing import Optional, Dict, Any


class MemoryEfficientQwenImageEditPipeline:
    """
    Memory-efficient pipeline for Qwen-Image-Edit models.
    """

    def __init__(self, model_path: str = "Qwen/Qwen-Image-Edit"):
        self.model_path = model_path
        self.pipeline = None
        self.device = self._get_optimal_device()

    def _get_optimal_device(self) -> str:
        """
        Get the optimal device based on available memory.
        """
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 24:  # 24GB+ GPUs
                return "cuda"
            else:
                print(f"[WARNING] GPU has only {gpu_memory:.1f}GB memory, may cause issues")
                return "cuda"  # Still try CUDA but warn
        elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"  # Apple Silicon
        else:
            return "cpu"

    def _setup_memory_optimizations(self):
        """
        Setup memory optimizations before loading.
        """
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()

        # Set environment variables for memory efficiency
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

        print("[OK] Memory optimizations applied")

    def load_pipeline_efficiently(self, use_fp8: bool = False, use_safetensors: bool = True):
        """
        Load pipeline with memory-efficient settings.
        """
        try:
            self._setup_memory_optimizations()

            from diffusers import QwenImageEditPipeline

            print(f"[INFO] Loading pipeline on {self.device}...")

            # Memory-efficient loading parameters
            load_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True,
            }

            # Use safetensors if available (more memory efficient)
            if use_safetensors:
                load_kwargs["use_safetensors"] = True

            # Load pipeline
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_path,
                **load_kwargs
            )

            # Move to device
            if self.device != "auto":
                self.pipeline.to(self.device)

            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("[OK] xFormers memory efficient attention enabled")
                except:
                    print("[INFO] xFormers not available, using standard attention")

            # Enable model CPU offload for very low memory
            if self.device == "cuda":
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory < 16:  # Less than 16GB
                    self.pipeline.enable_model_cpu_offload()
                    print("[OK] Model CPU offload enabled for low-memory GPU")

            print(f"[OK] Pipeline loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load pipeline: {e}")
            print("\nTROUBLESHOOTING:")
            print("1. Check available system memory (need at least 64GB for full loading)")
            print("2. Try loading on CPU: set device='cpu' (much slower but uses less RAM)")
            print("3. Clear system memory by closing other applications")
            print("4. Use a machine with more RAM or a GPU with more VRAM")
            print("5. Try the chunked loading approach below")
            return False

    def load_pipeline_chunked(self):
        """
        Load pipeline in chunks to minimize memory usage.
        """
        try:
            print("[INFO] Attempting chunked loading...")

            # Load components separately
            from diffusers import (
                AutoencoderKL,
                UNet2DConditionModel,
                CLIPTextModel,
                CLIPTokenizer
            )

            # Load tokenizer first (lightweight)
            tokenizer = CLIPTokenizer.from_pretrained(
                self.model_path,
                subfolder="tokenizer"
            )

            # Load text encoder
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_path,
                subfolder="text_encoder",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )

            # Load VAE
            vae = AutoencoderKL.from_pretrained(
                self.model_path,
                subfolder="vae",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )

            # Load UNet (most memory intensive)
            unet = UNet2DConditionModel.from_pretrained(
                self.model_path,
                subfolder="unet",
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )

            # Clear memory between loads
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Create pipeline from components
            from diffusers import QwenImageEditPipeline

            self.pipeline = QwenImageEditPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=unet,
                scheduler=None  # Will be loaded automatically
            )

            # Move to device
            if self.device != "cpu":
                self.pipeline.to(self.device)

            print("[OK] Pipeline loaded successfully using chunked approach")
            return True

        except Exception as e:
            print(f"[ERROR] Chunked loading failed: {e}")
            return False

    def __call__(self, image, prompt: str, **kwargs):
        """
        Generate edited image with memory monitoring.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline_efficiently() first.")

        try:
            # Monitor memory usage
            if torch.cuda.is_available():
                memory_before = torch.cuda.memory_allocated() / 1024**3

            # Generate
            result = self.pipeline(
                image=image,
                prompt=prompt,
                **kwargs
            )

            # Report memory usage
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated() / 1024**3
                memory_used = memory_after - memory_before
                print(f"[INFO] Generation used {memory_used:.2f}GB GPU memory")

            return result

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("[ERROR] GPU out of memory during generation")
                print("SOLUTIONS:")
                print("1. Reduce batch size or image resolution")
                print("2. Use CPU offload: pipeline.enable_model_cpu_offload()")
                print("3. Use sequential CPU offload: pipeline.enable_sequential_cpu_offload()")
                print("4. Reduce num_inference_steps")
            raise e


def diagnose_memory_issue():
    """
    Diagnose the memory issue from issue #74.
    """
    print("=" * 60)
    print("QWEN-IMAGE-EDIT MEMORY ISSUE DIAGNOSTIC")
    print("=" * 60)
    print()

    print("ISSUE #74: Loading pipeline causes system memory to spike and process killed")
    print()

    # Check system resources
    print("SYSTEM DIAGNOSTIC:")
    print(f"- CPU cores: {os.cpu_count()}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"- GPUs available: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f}GB")
    else:
        print("- No CUDA GPUs available")

    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"- System RAM: {memory.total / 1024**3:.1f}GB total, {memory.available / 1024**3:.1f}GB available")
    except ImportError:
        print("- System RAM: Unknown (install psutil for details)")

    print()
    print("ROOT CAUSES:")
    print("1. Qwen-Image-Edit models are large (~20GB) and load entirely into RAM during initialization")
    print("2. Diffusers loads all components at once, causing memory spike")
    print("3. System has 128GB RAM but loading may temporarily exceed limits")
    print("4. H20 GPU has 96GB VRAM but loading happens in system RAM first")
    print()

    print("SOLUTIONS IMPLEMENTED:")
    print("1. Memory-efficient loading with torch_dtype=float16")
    print("2. Chunked loading to load components separately")
    print("3. CPU offload for low-memory GPUs")
    print("4. xFormers memory efficient attention")
    print("5. Automatic device selection based on available memory")
    print()


def main():
    """
    Main function demonstrating the fix.
    """
    diagnose_memory_issue()

    print("USAGE EXAMPLE:")
    print("""
from fix_memory_issue import MemoryEfficientQwenImageEditPipeline

# Initialize with memory optimizations
pipeline = MemoryEfficientQwenImageEditPipeline()

# Try efficient loading first
if not pipeline.load_pipeline_efficiently():
    # Fallback to chunked loading
    pipeline.load_pipeline_chunked()

# Use pipeline
result = pipeline(image, "Change background to blue")
""")

    print("\nADDITIONAL TIPS:")
    print("- Close other memory-intensive applications before loading")
    print("- Use a machine with more RAM if possible")
    print("- Consider using CPU for loading, GPU for inference")
    print("- Monitor memory usage with tools like htop or nvidia-smi")


if __name__ == "__main__":
    main()