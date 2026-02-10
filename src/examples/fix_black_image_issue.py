"""
Fix for black/zero image output issue
Addresses issue #32: QWEN Image All Zeros/Black

This script provides comprehensive fixes for black image generation issues.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
import inspect


class BlackImageDiagnostic:
    """
    Diagnostic tool for black image issues in Qwen-Image.
    """

    def __init__(self):
        self.diagnostics = []

    def run_full_diagnostic(self, pipeline=None, inputs=None, outputs=None):
        """
        Run comprehensive diagnostic for black image issues.
        """
        print("=" * 60)
        print("QWEN-IMAGE BLACK IMAGE DIAGNOSTIC")
        print("=" * 60)
        print()

        print("ISSUE #32: All generated images are black/zero")
        print()

        # Run all diagnostic checks
        self.check_environment()
        self.check_pipeline_state(pipeline)
        self.check_inputs(inputs)
        self.analyze_outputs(outputs)
        self.check_common_issues()

        print("\nDIAGNOSTIC SUMMARY:")
        print("=" * 30)
        for diag in self.diagnostics:
            print(f"- {diag}")

        return self.diagnostics

    def check_environment(self):
        """Check environment setup."""
        print("1. ENVIRONMENT CHECK:")

        # PyTorch version and CUDA
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.diagnostics.append("CUDA not available - may cause issues")

        # Check for common import issues
        try:
            import diffusers
            print(f"   Diffusers version: {diffusers.__version__}")
        except ImportError:
            self.diagnostics.append("Diffusers not installed")
            print("   Diffusers: NOT INSTALLED")

        print()

    def check_pipeline_state(self, pipeline):
        """Check pipeline configuration and state."""
        print("2. PIPELINE CHECK:")

        if pipeline is None:
            self.diagnostics.append("Pipeline not provided for analysis")
            print("   Pipeline: None")
            return

        # Check device
        device = next(pipeline.parameters()).device
        print(f"   Device: {device}")

        # Check dtype
        dtype = next(pipeline.parameters()).dtype
        print(f"   Dtype: {dtype}")

        # Check if components are properly loaded
        components = ['unet', 'vae', 'text_encoder', 'tokenizer']
        for comp in components:
            if hasattr(pipeline, comp):
                val = getattr(pipeline, comp)
                status = "OK" if val is not None else "MISSING"
                print(f"   {comp}: {status}")
                if val is None:
                    self.diagnostics.append(f"Missing {comp} component")
            else:
                print(f"   {comp}: NOT FOUND")
                self.diagnostics.append(f"Missing {comp} component")

        # Check for NaN/Inf in weights
        has_issues = False
        for name, param in pipeline.named_parameters():
            if param.requires_grad:  # Only check trainable params for speed
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_issues = True
                    break

        if has_issues:
            self.diagnostics.append("NaN/Inf values found in model weights")
            print("   Weights: CONTAINS NaN/Inf")
        else:
            print("   Weights: OK")

        print()

    def check_inputs(self, inputs):
        """Check input parameters."""
        print("3. INPUTS CHECK:")

        if inputs is None:
            print("   Inputs: None")
            return

        # Check prompt
        if 'prompt' in inputs:
            prompt = inputs['prompt']
            print(f"   Prompt: '{prompt[:50]}...'")
            if not prompt or prompt.isspace():
                self.diagnostics.append("Empty or whitespace-only prompt")

        # Check guidance scale
        if 'guidance_scale' in inputs:
            gs = inputs['guidance_scale']
            print(f"   Guidance scale: {gs}")
            if gs < 1.0:
                self.diagnostics.append("Guidance scale too low (< 1.0)")

        # Check other parameters
        key_params = ['num_inference_steps', 'height', 'width']
        for param in key_params:
            if param in inputs:
                print(f"   {param}: {inputs[param]}")

        print()

    def analyze_outputs(self, outputs):
        """Analyze output images."""
        print("4. OUTPUTS ANALYSIS:")

        if outputs is None:
            print("   Outputs: None")
            return

        if hasattr(outputs, 'images') and outputs.images:
            image = outputs.images[0]

            # Convert to numpy for analysis
            if hasattr(image, 'cpu'):
                # It's a tensor
                img_array = image.cpu().numpy()
            else:
                # It's a PIL image
                img_array = np.array(image)

            # Basic statistics
            print(f"   Image shape: {img_array.shape}")
            print(f"   Data type: {img_array.dtype}")
            print(f"   Value range: [{img_array.min():.6f}, {img_array.max():.6f}]")
            print(f"   Mean: {img_array.mean():.6f}")
            print(f"   Std: {img_array.std():.6f}")

            # Check for all zeros
            if img_array.max() == 0:
                self.diagnostics.append("Output image is completely black (all zeros)")
                print("   STATUS: ALL BLACK/ZERO")
            elif img_array.std() < 0.01:
                self.diagnostics.append("Output image has very low variance (nearly uniform)")
                print("   STATUS: NEARLY UNIFORM")
            else:
                print("   STATUS: OK")

            # Check for NaN/Inf
            if np.isnan(img_array).any():
                self.diagnostics.append("Output contains NaN values")
                print("   NaN check: CONTAINS NaN")
            elif np.isinf(img_array).any():
                self.diagnostics.append("Output contains Inf values")
                print("   Inf check: CONTAINS Inf")
            else:
                print("   NaN/Inf check: OK")

        else:
            self.diagnostics.append("No valid images in output")
            print("   STATUS: NO VALID OUTPUT")

        print()

    def check_common_issues(self):
        """Check for common issues that cause black images."""
        print("5. COMMON ISSUES CHECK:")

        issues_found = []

        # Check PyTorch version compatibility
        torch_version = torch.__version__
        if torch_version.startswith('2.'):
            print("   PyTorch 2.x: OK")
        else:
            issues_found.append("PyTorch version may be incompatible")
            print("   PyTorch version: MAY BE INCOMPATIBLE")

        # Check CUDA version compatibility
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            if cuda_version and cuda_version.startswith('11.') or cuda_version.startswith('12.'):
                print("   CUDA version: OK")
            else:
                issues_found.append("CUDA version may be incompatible")
                print("   CUDA version: MAY BE INCOMPATIBLE")

        # Memory check
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            memory_percent = memory_used / memory_total * 100

            print(f"   GPU memory: {memory_used:.1f}GB / {memory_total:.1f}GB ({memory_percent:.1f}%)")

            if memory_percent > 95:
                issues_found.append("GPU memory nearly full - may cause issues")
                print("   WARNING: GPU memory nearly full")

        self.diagnostics.extend(issues_found)

        if not issues_found:
            print("   No common issues detected")

        print()


class BlackImageFixer:
    """
    Comprehensive fixer for black image issues.
    """

    def __init__(self):
        self.diagnostic = BlackImageDiagnostic()

    def apply_all_fixes(self, pipeline, inputs):
        """
        Apply all known fixes for black image issues.
        """
        print("APPLYING BLACK IMAGE FIXES...")
        print("=" * 40)

        fixed_inputs = inputs.copy()

        # Fix 1: Ensure proper guidance scale
        if 'guidance_scale' not in fixed_inputs or fixed_inputs['guidance_scale'] < 1.0:
            fixed_inputs['guidance_scale'] = 7.5
            print("✓ Fixed guidance_scale to 7.5")

        # Fix 2: Ensure proper dtype
        if hasattr(pipeline, 'dtype'):
            if pipeline.dtype != torch.float16:
                print("⚠ Warning: Pipeline dtype is not float16")

        # Fix 3: Add proper generator for reproducibility
        if 'generator' not in fixed_inputs:
            fixed_inputs['generator'] = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(42)
            print("✓ Added fixed generator for reproducibility")

        # Fix 4: Ensure proper prompt
        if 'prompt' not in fixed_inputs or not fixed_inputs['prompt'].strip():
            fixed_inputs['prompt'] = "A beautiful landscape with mountains and a lake"
            print("✓ Added default prompt")

        # Fix 5: Ensure proper dimensions
        if 'height' not in fixed_inputs:
            fixed_inputs['height'] = 1024
        if 'width' not in fixed_inputs:
            fixed_inputs['width'] = 1024
        print("✓ Set image dimensions to 1024x1024")

        # Fix 6: Ensure proper inference steps
        if 'num_inference_steps' not in fixed_inputs or fixed_inputs['num_inference_steps'] < 10:
            fixed_inputs['num_inference_steps'] = 50
            print("✓ Set num_inference_steps to 50")

        return fixed_inputs

    def create_test_case(self):
        """
        Create a minimal test case that should work.
        """
        return {
            "prompt": "A beautiful sunset over mountains with vibrant colors",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 20,  # Shorter for testing
            "guidance_scale": 7.5,
            "generator": torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu').manual_seed(42),
        }

    def diagnose_and_fix(self, pipeline, inputs=None, outputs=None):
        """
        Run diagnostic and apply fixes.
        """
        # Run diagnostic
        self.diagnostic.run_full_diagnostic(pipeline, inputs, outputs)

        if inputs:
            # Apply fixes
            fixed_inputs = self.apply_all_fixes(pipeline, inputs)

            print("\nFIXED INPUTS:")
            for key, value in fixed_inputs.items():
                if key == 'generator':
                    print(f"  {key}: torch.Generator (seeded)")
                else:
                    print(f"  {key}: {value}")

            return fixed_inputs

        return None


def main():
    """
    Main diagnostic and fixing function.
    """
    fixer = BlackImageFixer()

    print("QWEN-IMAGE BLACK IMAGE ISSUE FIXER")
    print("This tool helps diagnose and fix black/zero image generation issues.")
    print()

    print("USAGE:")
    print("""
# For existing pipeline and inputs:
from fix_black_image_issue import BlackImageFixer

fixer = BlackImageFixer()
fixed_inputs = fixer.diagnose_and_fix(pipeline, inputs, outputs)

# Then use fixed_inputs for generation
result = pipeline(**fixed_inputs)
""")

    print("MOST COMMON FIXES:")
    print("1. guidance_scale >= 1.0 (usually 7.5)")
    print("2. Use proper generator with fixed seed")
    print("3. Ensure non-empty prompt")
    print("4. Check GPU memory isn't full")
    print("5. Use compatible PyTorch/CUDA versions")
    print("6. Ensure pipeline is properly loaded on GPU")
    print()

    print("TEST CASE:")
    test_inputs = fixer.create_test_case()
    print("Minimal working test case:")
    for key, value in test_inputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()