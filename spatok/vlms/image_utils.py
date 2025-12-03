"""
Image encoding utilities for VLM APIs.

Supports multiple input formats:
- File paths (.bmp, .jpg, .jpeg, .png)
- Numpy arrays (both HWC and CHW formats)
- PIL Image objects

Handles both OpenCV (BGR) and PIL (RGB) channel orders.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from PIL import Image


def detect_array_format(array: np.ndarray) -> str:
    """
    Detect if numpy array is in HWC (Height, Width, Channel) or CHW (Channel, Height, Width) format.
    
    Args:
        array: Numpy array to analyze
        
    Returns:
        'HWC', 'CHW', or 'HW' (for grayscale without explicit channel dimension)
        
    Raises:
        ValueError: If array format cannot be determined
    """
    if array.ndim == 2:
        return 'HW'
    
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")
    
    shape = array.shape
    
    # Check if first dimension looks like channels (1, 3, or 4)
    if shape[0] in [1, 3, 4] and shape[0] < min(shape[1], shape[2]):
        return 'CHW'
    
    # Check if last dimension looks like channels
    if shape[2] in [1, 3, 4] and shape[2] < max(shape[0], shape[1]):
        return 'HWC'
    
    # If ambiguous, assume HWC (more common in PIL/scikit-image)
    if shape[2] in [1, 3, 4]:
        return 'HWC'
    
    raise ValueError(f"Cannot determine array format from shape {shape}")


def get_image_mime_type(format: str) -> str:
    """
    Get MIME type for different image formats.
    
    Args:
        format: Image format ('PNG', 'JPEG', 'JPG', 'BMP')
        
    Returns:
        MIME type string (e.g., 'image/png')
    """
    format_upper = format.upper()
    mime_types = {
        'PNG': 'image/png',
        'JPEG': 'image/jpeg',
        'JPG': 'image/jpeg',
        'BMP': 'image/bmp'
    }
    return mime_types.get(format_upper, 'image/png')


def normalize_array_to_hwc_rgb(
    array: np.ndarray,
    channel_order: str = 'RGB'
) -> np.ndarray:
    """
    Normalize numpy array to HWC format with RGB channel order.
    
    Args:
        array: Input numpy array
        channel_order: Current channel order ('RGB' or 'BGR')
        
    Returns:
        Numpy array in HWC format with RGB channel order
    """
    # Detect current format
    array_format = detect_array_format(array)
    
    # Convert CHW to HWC if needed
    if array_format == 'CHW':
        array = np.transpose(array, (1, 2, 0))
    elif array_format == 'HW':
        # Grayscale - add channel dimension
        array = np.expand_dims(array, axis=-1)
    
    # Convert BGR to RGB if needed
    if channel_order.upper() == 'BGR' and array.shape[-1] == 3:
        array = array[..., ::-1]  # Reverse the last dimension
    
    # Ensure uint8 dtype
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            # Assume normalized [0, 1] range
            array = (array * 255).astype(np.uint8)
        else:
            array = array.astype(np.uint8)
    
    return array


def encode_image_to_base64(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    format: str = 'PNG',
    channel_order: str = 'RGB'
) -> str:
    """
    Universal image encoder supporting multiple input types.
    
    Args:
        image_input: Can be:
            - str/Path: file path (.bmp, .jpg, .jpeg, .png)
            - np.ndarray: numpy array (supports both HWC and CHW)
            - PIL.Image: PIL Image object
        format: Output format for encoding ('PNG', 'JPEG', 'BMP')
        channel_order: For numpy arrays, specify 'RGB' or 'BGR' (default: 'RGB')
    
    Returns:
        Base64 encoded string of the image
        
    Raises:
        TypeError: If image_input type is not supported
        FileNotFoundError: If file path does not exist
        ValueError: If numpy array format is invalid
    """
    pil_image = None
    
    # Handle different input types
    if isinstance(image_input, (str, Path)):
        # Load from file path
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        pil_image = Image.open(image_path)
        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if pil_image.mode not in ['RGB', 'L']:
            pil_image = pil_image.convert('RGB')
    
    elif isinstance(image_input, np.ndarray):
        # Handle numpy array
        normalized_array = normalize_array_to_hwc_rgb(image_input, channel_order)
        
        # Handle grayscale (single channel)
        if normalized_array.shape[-1] == 1:
            normalized_array = normalized_array.squeeze(-1)
            pil_image = Image.fromarray(normalized_array, mode='L')
        else:
            pil_image = Image.fromarray(normalized_array, mode='RGB')
    
    elif isinstance(image_input, Image.Image):
        # Already PIL Image
        pil_image = image_input
        # Convert to RGB if needed
        if pil_image.mode not in ['RGB', 'L']:
            pil_image = pil_image.convert('RGB')
    
    else:
        raise TypeError(
            f"Unsupported image_input type: {type(image_input)}. "
            "Expected str, Path, np.ndarray, or PIL.Image.Image"
        )
    
    # Encode to base64
    buffered = BytesIO()
    save_format = format.upper()
    if save_format == 'JPG':
        save_format = 'JPEG'
    
    pil_image.save(buffered, format=save_format)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_base64


def get_image_data_uri(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    format: str = 'PNG',
    channel_order: str = 'RGB'
) -> str:
    """
    Get data URI for an image (base64 with MIME type prefix).
    
    Args:
        image_input: Image in any supported format
        format: Output format for encoding
        channel_order: Channel order for numpy arrays
        
    Returns:
        Data URI string (e.g., 'data:image/png;base64,iVBORw0KGgo...')
    """
    base64_str = encode_image_to_base64(image_input, format, channel_order)
    mime_type = get_image_mime_type(format)
    return f"data:{mime_type};base64,{base64_str}"
