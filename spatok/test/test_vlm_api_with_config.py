"""
Example usage of VLM API wrappers with configuration management.
"""

from spatok.vlms.vlm_api import OpenAIVLM, GeminiVLM, ClaudeVLM, GrokVLM
from spatok.vlms.config import get_config


def main():
    # Load configuration (tries .env, environment variables, ~/.vlm_config.json)
    config = get_config()
    
    print("VLM Configuration:", config)
    print()
    
    # Example 1: Using OpenAI
    openai_key = config.get_openai_key()
    if openai_key:
        print("Testing OpenAI GPT-4o...")
        vlm = OpenAIVLM(api_key=openai_key, model_name="gpt-4o")
        response = vlm.call(
            text_prompt="What is the capital of France?",
            image_input=None  # Text-only example
        )
        print(f"Response: {response}\n")
    else:
        print("OpenAI API key not configured\n")
    
    # Example 2: Using Claude with image
    anthropic_key = config.get_anthropic_key()
    if anthropic_key:
        print("Testing Claude 3.5 Sonnet...")
        vlm = ClaudeVLM(api_key=anthropic_key)
        # Assuming you have an image file
        # response = vlm.call(
        #     text_prompt="What's in this image?",
        #     image_input="path/to/image.jpg"
        # )
        # print(f"Response: {response}\n")
        print("(Skipping image example)\n")
    else:
        print("Anthropic API key not configured\n")
    
    # Example 3: Using Gemini
    google_key = config.get_google_key()
    if google_key:
        print("Testing Gemini 1.5 Pro...")
        vlm = GeminiVLM(api_key=google_key, model_name="gemini-1.5-pro")
        response = vlm.call(
            text_prompt="Explain quantum computing in one sentence."
        )
        print(f"Response: {response}\n")
    else:
        print("Google API key not configured\n")


if __name__ == "__main__":
    # First time setup (optional):
    # 1. Copy .env.example to .env and add your keys
    # 2. Or run: config.create_template_env() to create .env template
    # 3. Or set environment variables in your shell
    
    main()
