"""
Unit tests for locally-deployed VLM wrappers.

Tests the vlm_local.py module containing open-source VLM models.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from spatok.vlms.vlm_local import (
    Qwen2VLM, Qwen25VLM, Qwen3VLM,
    Qwen25Omni, Qwen3Omni,
    LLaVAVLM, InternVLM, 
    Phi3VisionVLM, GLM4VVLM, GLM45VVLM, CogVLM2VLM,
    Llama32VisionVLM
)


def generate_noise_image(width=224, height=224, seed=42):
    """
    Generate an artificial noise image as numpy array.
    
    Args:
        width: Image width
        height: Image height
        seed: Random seed for reproducibility
        
    Returns:
        Numpy array in HWC format (RGB, uint8)
    """
    np.random.seed(seed)
    # Generate random noise image (H, W, C) in RGB format
    noise_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return noise_image


def test_qwen2_vlm(image_input, cache_dir=None, device="cuda"):
    """Test Qwen2-VL wrapper."""
    print("Testing Qwen2-VL...")
    
    # Example with Qwen2-VL
    qwen_vlm = Qwen2VLM(model_path="Qwen/Qwen2-VL-7B-Instruct", cache_dir=cache_dir, device=device)
    response = qwen_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("Qwen2-VL:", response)


def test_qwen25_vlm(image_input, cache_dir=None, device="cuda"):
    """Test Qwen2.5-VL wrapper."""
    print("Testing Qwen2.5-VL...")
    
    # Example with Qwen2.5-VL
    qwen_vlm = Qwen25VLM(model_path="Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_dir, device=device)
    response = qwen_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("Qwen2.5-VL:", response)


def test_qwen3_vlm(image_input, cache_dir=None, device="cuda"):
    """Test Qwen3-VL wrapper."""
    print("Testing Qwen3-VL...")
    
    # Example with Qwen3-VL (latest release)
    qwen_vlm = Qwen3VLM(model_path="Qwen/Qwen3-VL-8B-Instruct", cache_dir=cache_dir, device=device)
    response = qwen_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("Qwen3-VL:", response)


def test_qwen25_omni(image_input, cache_dir=None, device="cuda"):
    """Test Qwen2.5-Omni wrapper."""
    print("Testing Qwen2.5-Omni...")
    
    # Example with Qwen2.5-Omni (multimodal)
    omni_vlm = Qwen25Omni(cache_dir=cache_dir, device=device)
    response = omni_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("Qwen2.5-Omni:", response)


def test_qwen3_omni(image_input, cache_dir=None, device="cuda"):
    """Test Qwen3-Omni wrapper."""
    print("Testing Qwen3-Omni...")
    
    # Example with Qwen3-Omni (multimodal MoE)
    omni_vlm = Qwen3Omni(cache_dir=cache_dir, device=device)
    response = omni_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("Qwen3-Omni:", response)


def test_llava_vlm(image_input, cache_dir=None, device="cuda"):
    """Test LLaVA wrapper."""
    print("Testing LLaVA...")
    
    # Example with LLaVA
    llava_vlm = LLaVAVLM(model_path="llava-hf/llava-1.5-7b-hf", cache_dir=cache_dir, device=device)
    response = llava_vlm.call(
        text_prompt="Describe this image in detail",
        image_input=image_input
    )
    print("LLaVA:", response)


def test_internvl(image_input, cache_dir=None, device="cuda"):
    """Test InternVL wrapper."""
    print("Testing InternVL...")
    
    # Example with InternVL
    intern_vlm = InternVLM(model_path="OpenGVLab/InternVL2-8B", cache_dir=cache_dir, device=device)
    response = intern_vlm.call(
        text_prompt="What objects are in this image?",
        image_input=image_input
    )
    print("InternVL:", response)


def test_phi3_vision(image_input, cache_dir=None, device="cuda"):
    """Test Phi-3 Vision wrapper."""
    print("Testing Phi-3 Vision...")
    
    # Example with Phi-3 Vision
    phi3_vlm = Phi3VisionVLM(cache_dir=cache_dir, device=device)
    response = phi3_vlm.call(
        text_prompt="Analyze this image",
        image_input=image_input
    )
    print("Phi-3 Vision:", response)


def test_glm4v_vlm(image_input, cache_dir=None, device="cuda"):
    """Test GLM-4V wrapper."""
    print("Testing GLM-4V...")
    
    glm_vlm = GLM4VVLM(cache_dir=cache_dir, device=device)
    response = glm_vlm.call(
        text_prompt="What do you see?",
        image_input=image_input
    )
    print("GLM-4V:", response)


def test_glm45v_vlm(image_input, cache_dir=None, device="cuda"):
    """Test GLM-4.5V wrapper."""
    print("Testing GLM-4.5V...")
    
    glm_vlm = GLM45VVLM(cache_dir=cache_dir, device=device)
    response = glm_vlm.call(
        text_prompt="What do you see?",
        image_input=image_input
    )
    print("GLM-4.5V:", response)


def test_cogvlm2_vlm(image_input, cache_dir=None, device="cuda"):
    """Test CogVLM2 wrapper."""
    print("Testing CogVLM2...")
    
    cog_vlm = CogVLM2VLM(cache_dir=cache_dir, device=device)
    response = cog_vlm.call(
        text_prompt="What do you see?",
        image_input=image_input
    )
    print("CogVLM2:", response)


def test_llama32_vision(image_input, cache_dir=None, device="cuda"):
    """Test Llama-3.2-Vision wrapper."""
    print("Testing Llama-3.2-Vision...")
    
    llama_vlm = Llama32VisionVLM(cache_dir=cache_dir, device=device)
    response = llama_vlm.call(
        text_prompt="What do you see in this image?",
        image_input=image_input
    )
    print("Llama-3.2-Vision:", response)


def test_numpy_input(image_input, cache_dir=None, device="cuda"):
    """Test VLM with numpy array input (OpenCV format)."""
    print("Testing numpy array input...")
    
    import cv2
    
    # Check if image_input is already a numpy array (from noise mode)
    if isinstance(image_input, np.ndarray):
        img_bgr = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    else:
        # Load from file path
        img_bgr = cv2.imread(str(image_input))  # BGR format, HWC
    
    qwen_vlm = Qwen3VLM(model_path="Qwen/Qwen3-VL-8B-Instruct", cache_dir=cache_dir, device=device)
    response = qwen_vlm.call(
        text_prompt="What's in this image?",
        image_input=img_bgr  # Will be auto-converted to RGB
    )
    print("Qwen3-VL (numpy):", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test locally-deployed VLM models")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "noise"],
        required=True,
        help="Test mode: 'image' to use an image file, 'noise' to use artificial noise"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to image file (required if mode='image')"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs='+',
        default=["qwen25omni", "qwen3omni", "internvl", "phi3", "minicpm"],
        help=(
            "Which model(s) to test (default: qwen3). "
            "Use 'all' to test all models, or specify multiple models: "
            "--model qwen2 qwen25 llava"
        )
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Width of noise image (default: 224)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Height of noise image (default: 224)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise generation (default: 42)"
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default='/data/miao/checkpoints/spatok_hf_cache',
        help="Directory to cache downloaded models (default: uses HF_HOME or ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run models on: 'cuda' (GPU) or 'cpu' (default: cuda). Use 'cpu' for models that cause OOM."
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID to use (e.g., 0, 1, 2). If not specified, uses default GPU (0)."
    )
    
    args = parser.parse_args()
    
    # Set GPU ID if specified
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        print(f"Using GPU {args.gpu_id}")
    
    # Setup cache directory
    cache_dir = None
    if args.model_cache_dir:
        cache_dir = Path(args.model_cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Set HuggingFace cache environment variable
        os.environ['HF_HOME'] = str(cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir / 'hub')
        print(f"Model cache directory: {cache_dir}")
        print(f"Models will be saved to: {cache_dir}/hub/models--<org>--<model-name>/")
    else:
        # Use default HuggingFace cache
        default_cache = Path.home() / '.cache' / 'huggingface'
        print(f"Using default HuggingFace cache: {default_cache}")
        print("To specify a custom cache directory, use --model_cache_dir")
    
    # Prepare image input based on mode
    if args.mode == "image":
        if args.image_path is None:
            parser.error("--image_path is required when mode='image'")
        image_input = args.image_path
        print(f"Using image file: {args.image_path}")
    else:  # mode == "noise"
        image_input = generate_noise_image(
            width=args.width,
            height=args.height,
            seed=args.seed
        )
        print(f"Generated noise image: {args.width}x{args.height}, seed={args.seed}")
    
    # Print test info
    print("="*60)
    print("Local VLM Tests")
    print("="*60)
    print(f"Test mode: {args.mode}")
    print(f"Model(s): {', '.join(args.model)}")
    print(f"Device: {args.device}")
    print("="*60)
    print("NOTE: First run may take time to download models from HuggingFace.")
    print("Ensure you have sufficient disk space and GPU memory.")
    if cache_dir:
        print(f"Models will be cached in: {cache_dir}/hub/")
    print("=" * 60)
    print()
    
    # Run selected test
    test_functions = {
        "qwen2": test_qwen2_vlm, # passed
        "qwen25": test_qwen25_vlm, # passed 
        "qwen3": test_qwen3_vlm, # passed
        "qwen25_omni": test_qwen25_omni, # not passed due to torch version issues
        "qwen3_omni": test_qwen3_omni, # not passed due to CUDA OOM 
        "llava": test_llava_vlm, # passed
        "internvl": test_internvl, # passed
        "phi3": test_phi3_vision, # not passed due to transformers version issues and CUDA OOM
        "glm4v": test_glm4v_vlm, # not passed due to CUDA OOM
        "glm45v": test_glm45v_vlm, # not passed due to CUDA OOM
        "cogvlm2": test_cogvlm2_vlm, # not passed due to CUDA OOM
        "llama32": test_llama32_vision, # not passed due to CUDA OOM
        "numpy": test_numpy_input # passed on Qwen3-VL
    }
    
    # Validate model names
    valid_models = set(test_functions.keys())
    models_to_test = []
    
    if "all" in args.model:
        models_to_test = list(test_functions.keys())
    else:
        for model_name in args.model:
            if model_name not in valid_models:
                parser.error(f"Invalid model: {model_name}. Choose from: {', '.join(sorted(valid_models))} or 'all'")
            models_to_test.append(model_name)
    
    # Test selected models
    if len(models_to_test) > 1:
        print(f"Testing {len(models_to_test)} models...\n")
        for model_name in models_to_test:
            test_function = test_functions[model_name]
            print("\n" + "=" * 60)
            print(f"Testing: {model_name}")
            print("="*60)
            try:
                test_function(image_input, cache_dir=cache_dir, device=args.device)
                print(f"✓ {model_name} test completed successfully")
            except Exception as e:
                print(f"✗ {model_name} test failed with error: {e}")
            print("=" * 60)
    else:
        # Test single model
        model_name = models_to_test[0]
        test_function = test_functions[model_name]
        test_function(image_input, cache_dir=cache_dir, device=args.device)
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
