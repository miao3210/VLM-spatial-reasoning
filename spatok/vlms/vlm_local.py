"""
VLM (Vision-Language Model) wrappers for locally-deployable models.

This module contains classes for open-source VLMs that can be run locally:
- Qwen2-VL (Alibaba) - Qwen2VLM
- Qwen2.5-VL (Alibaba) - Qwen25VLM
- Qwen3-VL (Alibaba) - Qwen3VLM (latest, released Dec 2024)
- Qwen2.5-Omni (Alibaba multimodal) - Qwen25OmniVLM
- Qwen3-Omni (Alibaba multimodal) - Qwen3OmniVLM
- LLaVA Series (LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision) - LLaVAVLM
- InternVL2 / InternVL2.5 (OpenGVLab) - InternVLM
- GLM-4V (Zhipu AI) - GLM4VVLM
- GLM-4.5V (Zhipu AI) - GLM45VVLM
- Phi-3 Vision (Microsoft) - Phi3VisionVLM
- GLM-4V (Zhipu AI) - GLM4VVLM
- CogVLM2 (Zhipu AI) - CogVLM2VLM
- Llama-3.2-Vision (Meta) - Llama32VisionVLM

All classes follow a consistent interface matching the API-based VLMs.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import functools

from spatok.vlms.image_utils import normalize_array_to_hwc_rgb


class BaseLocalVLM(ABC):
    """Abstract base class for locally-deployed VLM wrappers."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize local VLM.
        
        Args:
            model_path: Path or HuggingFace model ID
            device: Device to run on ('cuda', 'cpu', 'mps')
            torch_dtype: Torch dtype for model weights
            cache_dir: Directory to cache model files. If None, uses HuggingFace default.
                      Models are saved to: cache_dir/hub/models--<org>--<model-name>/
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
    
    @abstractmethod
    def load_model(self):
        """Load model and processor."""
        pass
    
    @abstractmethod
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        **kwargs
    ) -> str:
        """
        Generate response from VLM.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image (file path, numpy array, or PIL Image)
            **kwargs: Additional generation parameters
            
        Returns:
            Model response as string
        """
        pass
    
    def _prepare_image(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Image.Image:
        """
        Convert various image inputs to PIL Image.
        
        Args:
            image_input: Image in various formats
            
        Returns:
            PIL Image
        """
        if isinstance(image_input, (str, Path)):
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            normalized = normalize_array_to_hwc_rgb(image_input)
            return Image.fromarray(normalized).convert('RGB')
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")


class Qwen2VLM(BaseLocalVLM):
    """Qwen2-VL local model wrapper."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen2-VL.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Options: 'Qwen/Qwen2-VL-7B-Instruct', 'Qwen/Qwen2-VL-72B-Instruct'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load Qwen2-VL model and processor."""
        from transformers import Qwen2VLForConditionalGeneration
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen2-VL.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build conversation
        if image_input is None:
            # Text-only
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            # Image + text
            pil_image = self._prepare_image(image_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        
        # Apply chat template and process together
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process text and images together
        if image_input is not None:
            pil_image = self._prepare_image(image_input)
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response


class Qwen25VLM(BaseLocalVLM):
    """Qwen2.5-VL local model wrapper."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen2.5-VL.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Options: 'Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load Qwen2.5-VL model and processor."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen2.5-VL.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build conversation
        if image_input is None:
            # Text-only
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            # Image + text
            pil_image = self._prepare_image(image_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        
        # Apply chat template and process together
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process text and images together
        if image_input is not None:
            pil_image = self._prepare_image(image_input)
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response


class Qwen3VLM(BaseLocalVLM):
    """Qwen3-VL local model wrapper (latest Qwen VL series)."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen3-VL.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Options: 'Qwen/Qwen3-VL-8B-Instruct', 'Qwen/Qwen3-VL-70B-Instruct'
                       Note: Qwen3-VL has been released as of Dec 2024. For older versions,
                             use Qwen2VLM class for Qwen2-VL or Qwen2.5-VL models.
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load Qwen3-VL model and processor."""
        from transformers import Qwen3VLForConditionalGeneration
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen3-VL.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build conversation
        if image_input is None:
            # Text-only
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            # Image + text
            pil_image = self._prepare_image(image_input)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        
        # Apply chat template and process together
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process text and images together
        if image_input is not None:
            pil_image = self._prepare_image(image_input)
            inputs = self.processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response


class LLaVAVLM(BaseLocalVLM):
    """LLaVA series local model wrapper."""
    
    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize LLaVA model.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Options: 'llava-hf/llava-1.5-7b-hf', 'llava-hf/llava-1.5-13b-hf',
                               'llava-hf/llava-v1.6-mistral-7b-hf', 'llava-hf/llava-v1.6-34b-hf'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load LLaVA model and processor."""
        from transformers import LlavaForConditionalGeneration
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using LLaVA.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        if image_input is None:
            raise ValueError("LLaVA requires an image input")
        
        # Prepare image
        pil_image = self._prepare_image(image_input)
        
        # Build prompt (LLaVA format)
        prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs
            )
        
        # Decode (skip prompt)
        generated_text = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


class InternVLM(BaseLocalVLM):
    """InternVL2/InternVL2.5 local model wrapper."""
    
    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2-8B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize InternVL model.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Options: 'OpenGVLab/InternVL2-2B', 'OpenGVLab/InternVL2-8B',
                               'OpenGVLab/InternVL2-26B', 'OpenGVLab/InternVL2-40B'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
        
    def load_model(self):
        """Load InternVL model and processor with complete patching."""
        from transformers import AutoModel, AutoTokenizer, GenerationConfig
        from transformers.generation.utils import GenerationMixin
        import functools
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
            cache_dir=self.cache_dir
        ).eval()
        
        # PATCH: Fix language model for transformers v4.50+
        if hasattr(self.model, 'language_model'):
            lm = self.model.language_model
            
            # Step 1: Add GenerationMixin
            if not hasattr(lm, 'generate'):
                original_class = lm.__class__
                
                class PatchedLanguageModel(GenerationMixin, original_class):
                    pass
                
                lm.__class__ = PatchedLanguageModel
            
            # Step 2: Initialize generation_config
            if not hasattr(lm, 'generation_config') or lm.generation_config is None:
                if hasattr(lm, 'config'):
                    try:
                        lm.generation_config = GenerationConfig.from_model_config(lm.config)
                    except Exception:
                        lm.generation_config = GenerationConfig(
                            max_length=2048,
                            max_new_tokens=1024,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=getattr(lm.config, 'pad_token_id', 0),
                            eos_token_id=getattr(lm.config, 'eos_token_id', 2),
                        )
                else:
                    lm.generation_config = GenerationConfig()
        
            # Step 3: Patch forward method to handle new arguments
            original_forward = lm.forward
            
            @functools.wraps(original_forward)
            def patched_forward(*args, **kwargs):
                # Remove arguments not supported by older model code
                kwargs.pop('cache_position', None)
                return original_forward(*args, **kwargs)
            
            lm.forward = patched_forward
            
            # Step 4: Patch prepare_inputs_for_generation to handle past_key_values properly
            if hasattr(lm, 'prepare_inputs_for_generation'):
                original_prepare = lm.prepare_inputs_for_generation
                
                @functools.wraps(original_prepare)
                def patched_prepare_inputs(*args, **kwargs):
                    # Convert Cache objects to None for compatibility with old remote code
                    past_key_values = kwargs.get('past_key_values', None)
                    if past_key_values is not None and hasattr(past_key_values, '__class__'):
                        # Check if it's a Cache object (has 'get_seq_length' or similar methods)
                        if hasattr(past_key_values, 'get_seq_length') or hasattr(past_key_values, 'get_usable_length'):
                            # It's a new Cache object but model expects None or tuple format
                            # Set to None so the old model code initializes it properly
                            kwargs['past_key_values'] = None
                    
                    return original_prepare(*args, **kwargs)
                
                lm.prepare_inputs_for_generation = patched_prepare_inputs
            
            print("✓ Fully patched language_model for transformers v4.50+ compatibility")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )   
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using InternVL.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build generation_config without use_cache to avoid conflicts
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'do_sample': temperature > 0,
        }
        if temperature > 0:
            generation_config['temperature'] = temperature
    
        # Filter out use_cache from kwargs if present to avoid duplicate argument error
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'use_cache'}
        generation_config.update(filtered_kwargs)
    
        if image_input is None:
            response = self.model.chat(
                self.tokenizer,
                pixel_values=None,
                question=text_prompt,
                generation_config=generation_config
            )
        else:
            pil_image = self._prepare_image(image_input)
            
            # Access load_image from the model's module
            import sys
            model_module = sys.modules[self.model.__class__.__module__]
            
            if hasattr(model_module, 'load_image'):
                # Use the model's built-in load_image function
                pixel_values = model_module.load_image(
                    pil_image, 
                    max_num=12
                ).to(self.torch_dtype).to(self.device)
            else:
                # Fallback: manual preprocessing
                import torchvision.transforms as T
                from torchvision.transforms.functional import InterpolationMode
                
                IMAGENET_MEAN = (0.485, 0.456, 0.406)
                IMAGENET_STD = (0.229, 0.224, 0.225)
                
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                ])
                
                pixel_values = transform(pil_image).unsqueeze(0).to(
                    self.torch_dtype
                ).to(self.device)
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=text_prompt,
                generation_config=generation_config
            )
        
        return response


class Phi3VisionVLM(BaseLocalVLM):
    """Microsoft Phi-3 Vision local model wrapper."""
    
    def __init__(
        self,
        model_path: str = "microsoft/Phi-3-vision-128k-instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Phi-3 Vision model.
        
        Args:
            model_path: HuggingFace model ID or local path
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load Phi-3 Vision model and processor with flash-attention hard-disabled."""

        # Force eager attention via env so Transformers prefers non-flash paths.
        os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
        os.environ["DISABLE_FLASH_ATTN"] = "1"

        # Do NOT pre-insert flash_attn stubs: let importlib find_spec return None naturally

        # Monkeypatch Transformers' flash-attention checker to report unavailable
        try:
            import transformers.modeling_flash_attention_utils as fau
            import transformers.utils.import_utils as iu
            
            # Mark flash attention as unavailable
            if hasattr(fau, 'is_flash_attn_2_available'):
                fau.is_flash_attn_2_available = lambda: False
            if hasattr(iu, 'is_flash_attn_2_available'):
                iu.is_flash_attn_2_available = lambda: False

            # Ensure _is_package_available("flash_attn") returns False without raising
            if hasattr(iu, '_is_package_available'):
                _orig_is_pkg_avail = iu._is_package_available
                def _is_package_available_patched(pkg_name):
                    try:
                        if pkg_name in {"flash_attn", "flash-attn", "flash_attn_2"}:
                            return False
                        return _orig_is_pkg_avail(pkg_name)
                    except Exception:
                        # Treat any probe error as unavailable
                        return False
                iu._is_package_available = _is_package_available_patched
            
            # Patch the lazy import to return False/None instead of importing
            def _no_flash_lazy_import(implementation):
                # Return dummy functions that won't be called
                return None, None, None, None
            
            if hasattr(fau, "_lazy_imports"):
                fau._lazy_imports = _no_flash_lazy_import
                
            # Also patch the main lazy_import_flash_attention function
            def _no_flash_import(implementation, force_import=False):
                # Just return without doing anything - attention will fall back to eager
                return
            
            if hasattr(fau, "lazy_import_flash_attention"):
                fau.lazy_import_flash_attention = _no_flash_import
            
            # Also ensure import_utils._is_package_available('flash_attn') returns False via find_spec(None)
            # Our stub with __spec__ = None already ensures this, so no extra patch needed.
                
        except Exception as e:
            print(f"Warning: Could not fully patch flash attention utils: {e}")

        # Disable PyTorch SDPA flash/mem-efficient backends
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass

        # Load config first and force attention implementation to 'eager'
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        # Ensure both public and internal fields point to eager
        try:
            setattr(cfg, 'attn_implementation', 'eager')
            setattr(cfg, '_attn_implementation_internal', 'eager')
        except Exception:
            pass

        # Load model with the patched config (avoid passing flash-attn flags)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=cfg,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
            cache_dir=self.cache_dir,
        )

        # Double-ensure config sticks to eager
        try:
            self.model.config.attn_implementation = "eager"
            if hasattr(self.model.config, "_attn_implementation_internal"):
                self.model.config._attn_implementation_internal = "eager"
        except Exception:
            pass

        # PATCH: Fix Phi-3 Vision for transformers v4.50+ cache compatibility
        # The model's forward method calls get_usable_length(), seen_tokens, and get_max_length() 
        # which don't exist in DynamicCache. We need to monkey-patch DynamicCache to add these methods/attributes
        
        try:
            from transformers.cache_utils import DynamicCache
            
            # Add get_usable_length as an alias to get_seq_length if it doesn't exist
            if not hasattr(DynamicCache, 'get_usable_length'):
                def get_usable_length(self, *args, **kwargs):
                    # Map get_usable_length to get_seq_length for compatibility
                    return self.get_seq_length(*args, **kwargs)
                
                DynamicCache.get_usable_length = get_usable_length
                print("✓ Patched DynamicCache.get_usable_length -> get_seq_length")
            
            # Add get_max_length method if it doesn't exist
            if not hasattr(DynamicCache, 'get_max_length'):
                def get_max_length(self):
                    # Return None to indicate unlimited cache (DynamicCache has no max length)
                    # This is the expected behavior for dynamic caches
                    return None
                
                DynamicCache.get_max_length = get_max_length
                print("✓ Patched DynamicCache.get_max_length -> None")
            
            # Add seen_tokens as a property if it doesn't exist
            if not hasattr(DynamicCache, 'seen_tokens'):
                @property
                def seen_tokens(self):
                    # In DynamicCache, _seen_tokens tracks the number of tokens processed
                    # Fall back to get_seq_length() if _seen_tokens doesn't exist
                    if hasattr(self, '_seen_tokens'):
                        return self._seen_tokens
                    # Otherwise compute from the cache length
                    return self.get_seq_length()
                
                DynamicCache.seen_tokens = seen_tokens
                print("✓ Patched DynamicCache.seen_tokens property")
                
        except Exception as e:
            print(f"Warning: Could not patch DynamicCache: {e}")
        
        print("✓ Patched Phi-3 Vision for transformers v4.50+ cache compatibility")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Phi-3 Vision.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build messages
        if image_input is None:
            messages = [
                {"role": "user", "content": text_prompt}
            ]
            images = None
        else:
            pil_image = self._prepare_image(image_input)
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{text_prompt}"}
            ]
            images = [pil_image]
        
        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            prompt,
            images=images,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                use_cache=False,  # Disable cache to avoid compatibility issues with transformers v4.50+
                **kwargs
            )
        
        # Decode
        generated_text = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


class GLM4VVLM(BaseLocalVLM):
    """GLM-4V local model wrapper (Zhipu AI)."""
    
    def __init__(
        self,
        model_path: str = "THUDM/glm-4v-9b",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize GLM-4V model.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Default: 'THUDM/glm-4v-9b'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load GLM-4V model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
            cache_dir=self.cache_dir
        ).eval()
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        **kwargs
    ) -> str:
        """
        Generate response using GLM-4V.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build messages
        if image_input is None:
            messages = [{"role": "user", "content": text_prompt}]
        else:
            pil_image = self._prepare_image(image_input)
            # GLM-4V expects image as a direct property, not nested in content
            messages = [
                {
                    "role": "user",
                    "image": pil_image,
                    "content": text_prompt
                }
            ]
        
        # Prepare inputs
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)
        
        # Generate
        # Use max_length instead of max_new_tokens for GLM-4V compatibility
        # Also disable cache to avoid AttributeError with ChatGLMConfig
        gen_kwargs = {
            "max_length": inputs['input_ids'].shape[1] + max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "use_cache": False,  # Disable cache to avoid config compatibility issues
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response


class GLM45VVLM(BaseLocalVLM):
    """GLM-4.5V local model wrapper (Zhipu AI)."""
    
    def __init__(
        self,
        model_path: str = "zai-org/GLM-4.5V",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize GLM-4.5V model.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Default: 'zai-org/GLM-4.5V'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load GLM-4.5V model and processor."""
        from transformers import AutoProcessor, Glm4vMoeForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        self.model = Glm4vMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
            cache_dir=self.cache_dir
        ).eval()
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.8,
        enable_thinking: bool = False,
        **kwargs
    ) -> str:
        """
        Generate response using GLM-4.5V.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate (default: 8192)
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            enable_thinking: Enable thinking mode for deeper reasoning
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Build messages
        if image_input is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            pil_image = self._prepare_image(image_input)
            # GLM-4.5V expects structured content with type annotations
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        
        # Prepare inputs with processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            chat_template_kwargs={"enable_thinking": enable_thinking}
        ).to(self.model.device)
        
        # Remove token_type_ids if present (not needed for generation)
        inputs.pop("token_type_ids", None)
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            # Decode only the generated part
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False
            )
        
        return output_text


class CogVLM2VLM(BaseLocalVLM):
    """CogVLM2 local model wrapper (Zhipu AI)."""
    
    def __init__(
        self,
        model_path: str = "THUDM/cogvlm2-llama3-chat-19B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize CogVLM2 model.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Default: 'THUDM/cogvlm2-llama3-chat-19B'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load CogVLM2 model and tokenizer with flash attention disabled."""
        import sys
        from types import ModuleType
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        # Force disable flash attention via environment variables
        os.environ["DISABLE_FLASH_ATTN"] = "1"
        os.environ["XFORMERS_DISABLED"] = "1"
        
        # Create mock modules for flash_attn and xformers to prevent import errors
        # This is needed because CogVLM2's visual.py has hardcoded imports
        
        # Mock flash_attn module
        if 'flash_attn' not in sys.modules:
            flash_attn_mock = ModuleType('flash_attn')
            flash_attn_mock.__path__ = []
            sys.modules['flash_attn'] = flash_attn_mock
            
            # Mock flash_attn.flash_attn_interface
            flash_attn_interface = ModuleType('flash_attn.flash_attn_interface')
            sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
        
        # Mock xformers.ops to return dummy operations
        if 'xformers.ops' not in sys.modules:
            xformers_mock = ModuleType('xformers')
            xformers_mock.__path__ = []
            sys.modules['xformers'] = xformers_mock
            
            xformers_ops = ModuleType('xformers.ops')
            # Add dummy memory_efficient_attention that falls back to standard attention
            def dummy_memory_efficient_attention(*args, **kwargs):
                # Fallback to standard PyTorch attention
                import torch.nn.functional as F
                query, key, value = args[0], args[1], args[2]
                return F.scaled_dot_product_attention(query, key, value)
            
            xformers_ops.memory_efficient_attention = dummy_memory_efficient_attention
            sys.modules['xformers.ops'] = xformers_ops
        
        # Load config and force eager attention
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        # Set attention implementation to eager
        if hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'
        if hasattr(config, 'attn_implementation'):
            config.attn_implementation = 'eager'
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.device,
            cache_dir=self.cache_dir,
            attn_implementation="eager"
        ).eval()
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate response using CogVLM2.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Prepare image
        pil_image = self._prepare_image(image_input) if image_input is not None else None
        
        # Build input
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=text_prompt,
            images=[pil_image] if pil_image is not None else None,
            template_version='chat'
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(self.model.device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(self.model.device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(self.model.device),
            'images': [[input_by_model['images'][0].to(self.model.device).to(self.torch_dtype)]] if pil_image is not None else None,
        }
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "top_p": top_p,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response


class Llama32VisionVLM(BaseLocalVLM):
    """Meta Llama-3.2-Vision local model wrapper."""
    
    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Llama-3.2-Vision model.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Options: 'meta-llama/Llama-3.2-11B-Vision-Instruct',
                               'meta-llama/Llama-3.2-90B-Vision-Instruct'
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load Llama-3.2-Vision model and processor."""
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir
        )
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Llama-3.2-Vision.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Prepare image
        pil_image = self._prepare_image(image_input) if image_input is not None else None
        
        # Build messages
        if pil_image is None:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": text_prompt}
                ]}
            ]
        else:
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]}
            ]
        
        # Process inputs
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            pil_image,
            input_text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        gen_kwargs.update(kwargs)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode (skip prompt)
        generated_text = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


class Qwen25Omni(BaseLocalVLM):
    """Qwen2.5-Omni multimodal model wrapper (audio + vision + text)."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-Omni-7B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen2.5-Omni.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Default: 'Qwen/Qwen2.5-Omni-7B'
                       Note: Qwen2.5-Omni supports audio, vision, and text inputs
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
        
    def load_model(self):
        """Load Qwen2.5-Omni model and processor."""
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        
        # Load model with safetensors (no issue here)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            device_map=self.device,
            cache_dir=self.cache_dir
        )
        
        # Patch torch.load to bypass the version check for processor loading
        original_torch_load = torch.load
        
        def patched_load(f, *args, **kwargs):
            # Remove weights_only parameter if present to avoid the version check
            kwargs.pop('weights_only', None)
            # Call original torch.load without weights_only
            return original_torch_load(f, *args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_load
        
        try:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir
            )
        finally:
            # Restore original torch.load
            torch.load = original_torch_load
    
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        audio_input: Optional[Union[str, Path]] = None,
        use_audio_in_video: bool = False,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen2.5-Omni.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            audio_input: Optional audio input (file path or URL)
            use_audio_in_video: Whether to use audio in video inputs
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Import qwen_omni_utils
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError("qwen_omni_utils not found. Install with: pip install qwen-omni-utils")
        
        # Build conversation content
        content = []
        
        if image_input is not None:
            pil_image = self._prepare_image(image_input)
            content.append({"type": "image", "image": pil_image})
        
        if audio_input is not None:
            content.append({"type": "audio", "audio": audio_input})
        
        content.append({"type": "text", "text": text_prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        # Generate
        with torch.no_grad():
            text_ids, audio = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                use_audio_in_video=use_audio_in_video,
                **kwargs
            )
        
        # Decode response
        response = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response


class Qwen3Omni(BaseLocalVLM):
    """Qwen3-Omni multimodal MoE model wrapper (audio + vision + text)."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Qwen3-Omni.
        
        Args:
            model_path: HuggingFace model ID or local path
                       Default: 'Qwen/Qwen3-Omni-30B-A3B-Instruct'
                       Note: Qwen3-Omni supports audio, vision, and text inputs (MoE architecture)
            device: Device to run on
            torch_dtype: Dtype for model weights
            cache_dir: Directory to cache model files
        """
        super().__init__(model_path, device, torch_dtype, cache_dir)
        self.load_model()
    
    def load_model(self):
        """Load Qwen3-Omni model and processor."""
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            device_map=self.device,
            cache_dir=self.cache_dir,
            # Remove or comment out flash_attention_2 due to GLIBC issue
            # attn_implementation="flash_attention_2"
        )
        
        # Same patching approach for processor
        original_torch_load = torch.load
        
        def patched_load(f, *args, **kwargs):
            kwargs.pop('weights_only', None)
            return original_torch_load(f, *args, **kwargs)
        
        torch.load = patched_load
        
        try:
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir
            )
        finally:
            torch.load = original_torch_load
            
    def call(
        self,
        text_prompt: str,
        image_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
        audio_input: Optional[Union[str, Path]] = None,
        use_audio_in_video: bool = False,
        speaker: str = "Ethan",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate response using Qwen3-Omni.
        
        Args:
            text_prompt: Text prompt/question
            image_input: Optional image input
            audio_input: Optional audio input (file path or URL)
            use_audio_in_video: Whether to use audio in video inputs
            speaker: Speaker voice for audio generation (default: "Ethan")
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Model response text
        """
        # Import qwen_omni_utils
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError("qwen_omni_utils not found. Install with: pip install qwen-omni-utils")
        
        # Build conversation content
        content = []
        
        if image_input is not None:
            pil_image = self._prepare_image(image_input)
            content.append({"type": "image", "image": pil_image})
        
        if audio_input is not None:
            content.append({"type": "audio", "audio": audio_input})
        
        content.append({"type": "text", "text": text_prompt})
        
        conversation = [{"role": "user", "content": content}]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        # Generate
        with torch.no_grad():
            text_ids, audio = self.model.generate(
                **inputs,
                speaker=speaker,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                thinker_return_dict_in_generate=True,
                use_audio_in_video=use_audio_in_video,
                **kwargs
            )
        
        # Decode response
        response = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
