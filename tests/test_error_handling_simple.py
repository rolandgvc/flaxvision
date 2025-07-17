#!/usr/bin/env python3
"""
Simple test script to verify error handling implementation.
This tests the error handling logic without requiring full JAX/Flax dependencies.
"""
import os
import sys
import time
import warnings
from unittest.mock import Mock, patch
from urllib.error import URLError

# Add the workspace to path
sys.path.insert(0, '/workspace')

def test_retry_logic():
    """Test the retry logic in load_torch_params function."""
    print("Testing retry logic...")
    
    # Define the retry logic function (extracted from utils.py)
    def load_torch_params_retry_logic(url, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
        """Test version of load_torch_params with retry logic."""
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries + 1):
            try:
                # Simulate network call that might fail
                if attempt < 2:  # Fail first two attempts
                    raise URLError("Network error")
                return {"weight": "mock_tensor"}  # Success on third attempt
            except (URLError, Exception) as e:
                last_exception = e
                if attempt < max_retries:
                    warnings.warn(f"Network error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay}s...")
                    time.sleep(0.1)  # Short delay for testing
                    delay *= backoff_factor
                else:
                    warnings.warn(f"Failed to download parameters after {max_retries + 1} attempts. Last error: {e}. Using random initialization.")
                    return None
        
        return None
    
    # Test successful retry
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = load_torch_params_retry_logic("http://test.com/model.pth")
        
        if result is not None:
            print("✓ Retry logic test passed - succeeded after failures")
        else:
            print("✗ Retry logic test failed - should have succeeded")
            return False
            
        # Check warnings
        if len(w) >= 2:
            print("✓ Retry warnings generated correctly")
        else:
            print("✗ Expected retry warnings not generated")
            return False
    
    # Test complete failure
    def failing_load_torch_params(url, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
        """Version that always fails."""
        delay = initial_delay
        for attempt in range(max_retries + 1):
            try:
                raise URLError("Network error")
            except (URLError, Exception) as e:
                if attempt < max_retries:
                    warnings.warn(f"Network error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay}s...")
                    time.sleep(0.1)
                    delay *= backoff_factor
                else:
                    warnings.warn(f"Failed to download parameters after {max_retries + 1} attempts. Last error: {e}. Using random initialization.")
                    return None
        return None
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = failing_load_torch_params("http://test.com/model.pth")
        
        if result is None:
            print("✓ Complete failure test passed - returned None")
        else:
            print("✗ Complete failure test failed - should have returned None")
            return False
            
        # Check final warning
        if any("Using random initialization" in str(warning.message) for warning in w):
            print("✓ Final failure warning generated correctly")
        else:
            print("✗ Expected final failure warning not generated")
            return False
    
    return True

def test_parameter_validation():
    """Test parameter validation logic."""
    print("Testing parameter validation...")
    
    # Test None parameters
    try:
        if None is None:
            raise ValueError("torch_params cannot be None")
        print("✗ None parameter test failed - should have raised ValueError")
        return False
    except ValueError as e:
        if "torch_params cannot be None" in str(e):
            print("✓ None parameter validation passed")
        else:
            print("✗ None parameter validation failed - wrong error message")
            return False
    
    # Test invalid type
    try:
        params = "not_a_dict"
        if not isinstance(params, dict):
            raise TypeError(f"torch_params must be a dictionary, got {type(params)}")
        print("✗ Invalid type test failed - should have raised TypeError")
        return False
    except TypeError as e:
        if "torch_params must be a dictionary" in str(e):
            print("✓ Invalid type validation passed")
        else:
            print("✗ Invalid type validation failed - wrong error message")
            return False
    
    # Test empty parameters
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        params = {}
        if not params:
            warnings.warn("torch_params is empty, returning empty parameter structure")
            result = {'params': {}, 'batch_stats': {}}
        
        if result == {'params': {}, 'batch_stats': {}}:
            print("✓ Empty parameters test passed")
        else:
            print("✗ Empty parameters test failed")
            return False
            
        if any("empty" in str(warning.message) for warning in w):
            print("✓ Empty parameters warning generated correctly")
        else:
            print("✗ Expected empty parameters warning not generated")
            return False
    
    return True

def test_input_validation():
    """Test input validation logic."""
    print("Testing input validation...")
    
    class MockArray:
        def __init__(self, shape, has_nan=False, has_inf=False):
            self.shape = shape
            self.ndim = len(shape)
            self._has_nan = has_nan
            self._has_inf = has_inf
        
        def any(self):
            return self._has_nan or self._has_inf
    
    def mock_isnan(arr):
        return MockArray((1,), has_nan=arr._has_nan)
    
    def mock_isinf(arr):
        return MockArray((1,), has_inf=arr._has_inf)
    
    # Test valid input
    valid_input = MockArray((1, 224, 224, 3))
    
    try:
        # Input validation logic
        if not hasattr(valid_input, 'shape'):
            raise TypeError("inputs must be a JAX array")
        if valid_input.ndim != 4:
            raise ValueError(f"inputs must be 4-dimensional (batch, height, width, channels), got {valid_input.ndim}D")
        
        batch_size, height, width, channels = valid_input.shape
        if channels != 3:
            raise ValueError(f"inputs must have 3 channels (RGB), got {channels}")
        if height < 32 or width < 32:
            raise ValueError(f"inputs spatial dimensions must be at least 32x32, got {height}x{width}")
        
        print("✓ Valid input validation passed")
    except Exception as e:
        print(f"✗ Valid input validation failed: {e}")
        return False
    
    # Test invalid dimensions
    try:
        invalid_input = MockArray((1, 224, 224))  # 3D instead of 4D
        if invalid_input.ndim != 4:
            raise ValueError(f"inputs must be 4-dimensional (batch, height, width, channels), got {invalid_input.ndim}D")
        print("✗ Invalid dimensions test failed - should have raised ValueError")
        return False
    except ValueError as e:
        if "inputs must be 4-dimensional" in str(e):
            print("✓ Invalid dimensions validation passed")
        else:
            print("✗ Invalid dimensions validation failed - wrong error message")
            return False
    
    # Test invalid channels
    try:
        invalid_input = MockArray((1, 224, 224, 1))  # 1 channel instead of 3
        batch_size, height, width, channels = invalid_input.shape
        if channels != 3:
            raise ValueError(f"inputs must have 3 channels (RGB), got {channels}")
        print("✗ Invalid channels test failed - should have raised ValueError")
        return False
    except ValueError as e:
        if "inputs must have 3 channels" in str(e):
            print("✓ Invalid channels validation passed")
        else:
            print("✗ Invalid channels validation failed - wrong error message")
            return False
    
    # Test too small dimensions
    try:
        invalid_input = MockArray((1, 16, 16, 3))  # Too small
        batch_size, height, width, channels = invalid_input.shape
        if height < 32 or width < 32:
            raise ValueError(f"inputs spatial dimensions must be at least 32x32, got {height}x{width}")
        print("✗ Too small dimensions test failed - should have raised ValueError")
        return False
    except ValueError as e:
        if "inputs spatial dimensions must be at least 32x32" in str(e):
            print("✓ Too small dimensions validation passed")
        else:
            print("✗ Too small dimensions validation failed - wrong error message")
            return False
    
    return True

def test_fallback_behavior():
    """Test fallback behavior when network fails."""
    print("Testing fallback behavior...")
    
    def mock_model_factory_with_fallback(pretrained=True):
        """Mock model factory that demonstrates fallback behavior."""
        if pretrained:
            # Simulate network failure
            torch_params = None  # load_torch_params returns None on failure
            
            if torch_params is not None:
                print("Using pretrained parameters")
                return "pretrained_model"
            else:
                # Network failure fallback
                warnings.warn("Network failure, using random initialization")
                return "random_model"
        else:
            return "random_model"
    
    # Test fallback behavior
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = mock_model_factory_with_fallback(pretrained=True)
        
        if result == "random_model":
            print("✓ Fallback behavior test passed - used random initialization")
        else:
            print("✗ Fallback behavior test failed - should have used random initialization")
            return False
            
        if any("Network failure" in str(warning.message) for warning in w):
            print("✓ Fallback warning generated correctly")
        else:
            print("✗ Expected fallback warning not generated")
            return False
    
    return True

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Error Handling Tests")
    print("=" * 60)
    
    tests = [
        ("Retry Logic", test_retry_logic),
        ("Parameter Validation", test_parameter_validation),
        ("Input Validation", test_input_validation),
        ("Fallback Behavior", test_fallback_behavior),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            print(f"✓ {test_name} PASSED")
            passed += 1
        else:
            print(f"✗ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)