"""
Unit tests for image utility functions.

Tests the image_utils.py module for image encoding and format conversion.
"""

import numpy as np
from PIL import Image

from spatok.vlms.image_utils import (
    encode_image_to_base64,
    detect_array_format,
    normalize_array_to_hwc_rgb,
    get_image_mime_type,
    get_image_data_uri
)


def test_detect_array_format():
    """Test array format detection."""
    print("Testing array format detection...")
    
    # Test HWC format
    hwc_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    assert detect_array_format(hwc_array) == 'HWC', "Failed to detect HWC format"
    print("✓ HWC format detected correctly")
    
    # Test CHW format
    chw_array = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    assert detect_array_format(chw_array) == 'CHW', "Failed to detect CHW format"
    print("✓ CHW format detected correctly")
    
    # Test grayscale (HW)
    hw_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    assert detect_array_format(hw_array) == 'HW', "Failed to detect HW format"
    print("✓ HW format detected correctly")


def test_normalize_array():
    """Test array normalization to HWC RGB."""
    print("\nTesting array normalization...")
    
    # Test CHW to HWC conversion
    chw_array = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
    hwc_result = normalize_array_to_hwc_rgb(chw_array)
    assert hwc_result.shape == (224, 224, 3), "CHW to HWC conversion failed"
    print("✓ CHW to HWC conversion successful")
    
    # Test BGR to RGB conversion
    bgr_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    rgb_result = normalize_array_to_hwc_rgb(bgr_array, channel_order='BGR')
    # Check if channels were reversed
    np.testing.assert_array_equal(rgb_result[:, :, 0], bgr_array[:, :, 2])
    print("✓ BGR to RGB conversion successful")
    
    # Test grayscale expansion
    gray_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    result = normalize_array_to_hwc_rgb(gray_array)
    assert result.shape == (224, 224, 1), "Grayscale expansion failed"
    print("✓ Grayscale expansion successful")


def test_mime_types():
    """Test MIME type retrieval."""
    print("\nTesting MIME types...")
    
    assert get_image_mime_type('PNG') == 'image/png'
    assert get_image_mime_type('JPEG') == 'image/jpeg'
    assert get_image_mime_type('JPG') == 'image/jpeg'
    assert get_image_mime_type('BMP') == 'image/bmp'
    print("✓ All MIME types correct")


def test_base64_encoding():
    """Test base64 encoding for different input types."""
    print("\nTesting base64 encoding...")
    
    # Create a simple test image
    test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_image = Image.fromarray(test_array)
    
    # Test with PIL Image
    base64_str = encode_image_to_base64(test_image)
    assert isinstance(base64_str, str), "Base64 encoding failed for PIL Image"
    assert len(base64_str) > 0, "Base64 string is empty"
    print("✓ PIL Image encoding successful")
    
    # Test with numpy array
    base64_str = encode_image_to_base64(test_array)
    assert isinstance(base64_str, str), "Base64 encoding failed for numpy array"
    assert len(base64_str) > 0, "Base64 string is empty"
    print("✓ Numpy array encoding successful")


def test_data_uri():
    """Test data URI generation."""
    print("\nTesting data URI generation...")
    
    test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    data_uri = get_image_data_uri(test_array, format='PNG')
    
    assert data_uri.startswith('data:image/png;base64,'), "Data URI format incorrect"
    print("✓ Data URI generation successful")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Image Utils Tests")
    print("=" * 60)
    
    test_detect_array_format()
    test_normalize_array()
    test_mime_types()
    test_base64_encoding()
    test_data_uri()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
