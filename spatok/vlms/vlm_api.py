"""
VLM (Vision-Language Model) API wrappers for API-only providers.

This module contains classes for closed-source VLMs that are ONLY accessible via API:
- OpenAI (GPT-4V, GPT-4o, GPT-4o-mini)
- Google Gemini (Gemini 1.5 Pro/Flash, Gemini 2.0)
- Anthropic Claude (Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku)
- Grok/X.AI (Grok-2 Vision)
- Reka AI (Reka Core/Flash/Edge)

For locally-deployable open-source models, see vlm_local.py

All classes follow a consistent interface for easy provider switching.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image

from spatok.vlms.image_utils import encode_image_to_base64, get_image_mime_type, normalize_array_to_hwc_rgb


class BaseVLM(ABC):
    """Abstract base class for VLM API wrappers."""
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initialize VLM client.
        
        Args:
            api_key: API key for the service
            model_name: Model identifier
        """
        self.api_key = api_key
        self.model_name = model_name
    
    @abstractmethod
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        **kwargs
    ) -> str:
        """
        Call the VLM/LLM API.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image (file path, numpy array, or PIL Image)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Model response as string
        """
        pass


class OpenAIVLM(BaseVLM):
    """OpenAI GPT-4V/GPT-4o API wrapper."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        base_url: Optional[str] = None
    ):
        """
        Initialize OpenAI VLM client.
        
        Args:
            api_key: OpenAI API key
            model_name: Model name (e.g., 'gpt-4o', 'gpt-4-turbo', 'gpt-4-vision-preview')
            base_url: Optional custom base URL for API endpoint
        """
        super().__init__(api_key, model_name)
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Call OpenAI's VLM/LLM API.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Model response text
        """
        # Build message content
        if image_input is None:
            # Text-only request
            messages = [
                {"role": "user", "content": text_prompt}
            ]
        else:
            # Image + text request
            base64_image = encode_image_to_base64(image_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content


class GeminiVLM(BaseVLM):
    """Google Gemini API wrapper."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-pro"
    ):
        """
        Initialize Gemini VLM client.
        
        Args:
            api_key: Google API key
            model_name: Model name (e.g., 'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro-vision')
        """
        super().__init__(api_key, model_name)
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        **kwargs
    ) -> str:
        """
        Call Gemini's VLM/LLM API.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            temperature: Sampling temperature (0-2)
            max_output_tokens: Maximum tokens in response
            **kwargs: Additional generation config parameters
            
        Returns:
            Model response text
        """
        import google.generativeai as genai
        
        # Build generation config
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs
        )
        
        # Build content
        if image_input is None:
            # Text-only request
            content = [text_prompt]
        else:
            # Image + text request
            # Convert to PIL Image if needed
            if isinstance(image_input, (str, Path)):
                pil_image = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                normalized = normalize_array_to_hwc_rgb(image_input)
                pil_image = Image.fromarray(normalized)
            elif isinstance(image_input, Image.Image):
                pil_image = image_input
            else:
                raise TypeError(f"Unsupported image type: {type(image_input)}")
            
            content = [text_prompt, pil_image]
        
        # Make API call
        response = self.client.generate_content(
            content,
            generation_config=generation_config
        )
        
        return response.text


class ClaudeVLM(BaseVLM):
    """Anthropic Claude API wrapper."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-sonnet-4-5-20250929"
    ):
        """
        Initialize Claude VLM client.
        
        Args:
            api_key: Anthropic API key
            model_name: Model name (e.g., 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001', 
                       'claude-3-haiku-20240307')
        """
        super().__init__(api_key, model_name)
        from anthropic import Anthropic
        
        self.client = Anthropic(api_key=api_key)
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Call Claude's VLM/LLM API.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            **kwargs: Additional parameters for messages API
            
        Returns:
            Model response text
        """
        # Build message content
        if image_input is None:
            # Text-only request
            content = [
                {"type": "text", "text": text_prompt}
            ]
        else:
            # Image + text request
            base64_image = encode_image_to_base64(image_input, format='PNG')
            mime_type = get_image_mime_type('PNG')
            
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image
                    }
                },
                {"type": "text", "text": text_prompt}
            ]
        
        # Make API call
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": content}
            ],
            **kwargs
        )
        
        return response.content[0].text


class GrokVLM(BaseVLM):
    """X.AI Grok API wrapper."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "grok-2-vision-1212",
        base_url: str = "https://api.x.ai/v1"
    ):
        """
        Initialize Grok VLM client.
        
        Args:
            api_key: X.AI API key
            model_name: Model name (e.g., 'grok-2-vision-1212', 'grok-vision-beta')
            base_url: X.AI API base URL
        """
        super().__init__(api_key, model_name)
        from openai import OpenAI
        
        # Grok uses OpenAI-compatible API
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Call Grok's VLM/LLM API.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Model response text
        """
        # Build message content (same format as OpenAI)
        if image_input is None:
            # Text-only request
            messages = [
                {"role": "user", "content": text_prompt}
            ]
        else:
            # Image + text request
            base64_image = encode_image_to_base64(image_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response.choices[0].message.content


class RekaVLM(BaseVLM):
    """Reka AI API wrapper."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "reka-core",
        base_url: str = "https://api.reka.ai"
    ):
        """
        Initialize Reka VLM client.
        
        Args:
            api_key: Reka API key
            model_name: Model name (e.g., 'reka-core', 'reka-flash', 'reka-edge')
            base_url: Reka API base URL
        """
        super().__init__(api_key, model_name)
        import requests
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-Api-Key": api_key,
            "Content-Type": "application/json"
        })
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Call Reka's VLM/LLM API.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Model response text
        """
        # Build message content
        media_url = None
        if image_input is not None:
            # Encode image to base64 data URI
            base64_image = encode_image_to_base64(image_input, format='PNG')
            media_url = f"data:image/png;base64,{base64_image}"
        
        # Prepare request payload
        payload = {
            "model_name": self.model_name,
            "conversation_history": [
                {
                    "type": "human",
                    "text": text_prompt,
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Add media if present
        if media_url:
            payload["conversation_history"][0]["media_url"] = media_url
        
        # Make API call
        response = self.session.post(
            f"{self.base_url}/chat",
            json=payload
        )
        response.raise_for_status()
        
        return response.json()["text"]
