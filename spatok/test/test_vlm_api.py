"""
Unit tests for API-based VLM wrappers.

Tests the vlm_api.py module containing API-only VLM providers.
"""

import argparse
import numpy as np
from pathlib import Path
from spatok.vlms.vlm_api import OpenAIVLM, GeminiVLM, ClaudeVLM, GrokVLM, RekaVLM
from spatok.vlms.config import get_config


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


def test_openai_vlm(image_input, api_key=None):
    """Test OpenAI VLM wrapper."""
    print("Testing OpenAI VLM...")
    
    if api_key is None:
        api_key = get_config().get_openai_key()
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set via .env, environment variable, or config file.")
    
    openai_vlm = OpenAIVLM(api_key=api_key, model_name="gpt-4o")
    response = openai_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("OpenAI:", response)


def test_gemini_vlm(image_input, api_key=None):
    """Test Gemini VLM wrapper."""
    print("Testing Gemini VLM...")
    
    if api_key is None:
        api_key = get_config().get_google_key()
    
    if not api_key:
        raise ValueError("Google API key not found. Set via .env, environment variable, or config file.")
    
    # Try gemini-1.5-pro-latest or gemini-2.0-flash-exp for better compatibility
    gemini_vlm = GeminiVLM(api_key=api_key, model_name="gemini-1.5-flash-latest")
    response = gemini_vlm.call(
        text_prompt="Describe this image",
        image_input=image_input
    )
    print("Gemini:", response)


def test_claude_vlm(image_input, api_key=None):
    """Test Claude VLM wrapper."""
    print("Testing Claude VLM...")
    
    if api_key is None:
        api_key = get_config().get_anthropic_key()
    
    if not api_key:
        raise ValueError("Anthropic API key not found. Set via .env, environment variable, or config file.")
    
    claude_vlm = ClaudeVLM(api_key=api_key)
    response = claude_vlm.call(
        text_prompt="What do you see?",
        image_input=image_input
    )
    print("Claude:", response)


def test_grok_vlm(image_input, api_key=None):
    """Test Grok VLM wrapper."""
    print("Testing Grok VLM...")
    
    if api_key is None:
        api_key = get_config().get_xai_key()
    
    if not api_key:
        raise ValueError("X.AI API key not found. Set via .env, environment variable, or config file.")
    
    grok_vlm = GrokVLM(api_key=api_key)
    response = grok_vlm.call(
        text_prompt="Analyze this image",
        image_input=image_input
    )
    print("Grok:", response)


def test_reka_vlm(image_input, api_key=None):
    """Test Reka VLM wrapper."""
    print("Testing Reka VLM...")
    
    if api_key is None:
        api_key = get_config().get_reka_key()
    
    if not api_key:
        raise ValueError("Reka API key not found. Set via .env, environment variable, or config file.")
    
    reka_vlm = RekaVLM(api_key=api_key, model_name="reka-core")
    response = reka_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("Reka:", response)


def test_numpy_input(image_input, api_key=None):
    """Test VLM with numpy array input."""
    print("Testing numpy array input...")
    
    if api_key is None:
        api_key = get_config().get_openai_key()
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Set via .env, environment variable, or config file.")
    
    openai_vlm = OpenAIVLM(api_key=api_key, model_name="gpt-4o")
    response = openai_vlm.call(
        text_prompt="What's in this image?",
        image_input=image_input
    )
    print("OpenAI (numpy):", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test API-based VLM wrappers")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "noise"],
        default="noise",
        help="Test mode: 'image' (use real image) or 'noise' (generate random noise)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "gemini", "claude", "grok", "reka", "numpy", "all"],
        default="all",
        help="Which API provider to test (default: all)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to image file (required when mode='image')"
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
    
    args = parser.parse_args()
    
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
    print("=" * 60)
    print("API VLM Tests")
    print("=" * 60)
    print(f"Test mode: {args.mode}")
    print(f"Model: {args.model}")
    print("=" * 60)
    print("NOTE: Requires valid API keys in .env, environment variables, or ~/.vlm_config.json")
    print("=" * 60)
    print()
    
    # Run selected test
    test_functions = {
        "openai": test_openai_vlm, # passed 
        "gemini": test_gemini_vlm, # passed
        "claude": test_claude_vlm, # passed
        "grok": test_grok_vlm, # passed
        "reka": test_reka_vlm, # no api key yet
        "numpy": test_numpy_input # passed
    }
    
    if args.model == "all":
        # Test all models
        print("Testing all API providers...\n")
        for model_name, test_function in test_functions.items():
            print("\n" + "=" * 60)
            print(f"Testing: {model_name}")
            print("=" * 60)
            try:
                test_function(image_input)
                print(f"✓ {model_name} test completed successfully")
            except Exception as e:
                print(f"✗ {model_name} test failed with error: {e}")
            print("=" * 60)
    else:
        # Test single model
        test_function = test_functions[args.model]
        print(f"Testing {args.model}...")
        try:
            test_function(image_input)
            print(f"\n✓ Test completed successfully")
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
    
    print("\nTest completed!")

