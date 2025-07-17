#!/usr/bin/env python3
"""
Minimal test for tensor security functions without external dependencies
"""

import sys
import os
import unittest
import logging

# Add the current directory to the path to import flaxvision
sys.path.insert(0, os.path.dirname(__file__))

# Try importing with minimal dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Mock the missing dependencies and import our security functions
if not NUMPY_AVAILABLE:
    # Create a minimal numpy mock
    class MockNumpy:
        def isfinite(self, x):
            return MockArray([True] * len(x))
        def issubdtype(self, dtype, number):
            return True
        def number(self):
            return int
        def random(self):
            return MockRandom()
        def transpose(self, x, axes=None):
            return x
        def array(self, x):
            return MockArray(x)
        def inf(self):
            return float('inf')
        def nan(self):
            return float('nan')
    
    class MockArray:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data),) if isinstance(data, list) else (1,)
            self.ndim = 1
            self.dtype = 'float32'
            self.nbytes = len(data) * 4 if isinstance(data, list) else 4
        
        def all(self):
            return all(self.data) if isinstance(self.data, list) else bool(self.data)
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __setitem__(self, key, value):
            self.data[key] = value
    
    class MockRandom:
        def randn(self, *args):
            return MockArray([1.0] * (args[0] if args else 1))
    
    np = MockNumpy()
    np.random = MockRandom()

if not TORCH_AVAILABLE:
    # Create a minimal torch mock
    class MockTorch:
        def randn(self, *args):
            return MockTensor([1.0] * (args[0] if args else 1))
        def tensor(self, data):
            return MockTensor(data)
    
    class MockTensor:
        def __init__(self, data):
            self.data = data
            self.shape = (len(data),) if isinstance(data, list) else (1,)
        
        def detach(self):
            return self
        
        def numpy(self):
            return np.array(self.data)
    
    torch = MockTorch()

# Now import our security functions
from flaxvision.utils import (
    validate_tensor_security,
    validate_parameter_structure,
    TensorSecurityError,
    monitor_memory_usage
)

class TestTensorSecurityMinimal(unittest.TestCase):
    """Minimal tests for tensor security validation"""
    
    def test_tensor_security_validation_basic(self):
        """Test basic tensor security validation"""
        # Test with mock data
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        if NUMPY_AVAILABLE:
            # Test with real numpy
            tensor = np.array(test_data)
            result = validate_tensor_security(tensor)
            self.assertTrue(result is not None)
        
        # Test with our mock
        mock_tensor = np.array(test_data)
        result = validate_tensor_security(mock_tensor)
        self.assertTrue(result is not None)
    
    def test_parameter_structure_validation(self):
        """Test parameter structure validation"""
        # Test valid structure
        params = {
            'layer1': {'weight': [1.0, 2.0], 'bias': [0.1]},
            'layer2': {'weight': [3.0, 4.0], 'bias': [0.2]}
        }
        
        # Should not raise exception
        validate_parameter_structure(params)
        
        # Test invalid structure
        with self.assertRaises(TensorSecurityError):
            validate_parameter_structure("not a dict")
    
    def test_tensor_security_error_handling(self):
        """Test tensor security error handling"""
        # Test None tensor
        with self.assertRaises(TensorSecurityError):
            validate_tensor_security(None)
        
        # Test parameter validation with too many params
        large_params = {f'param_{i}': [1.0] for i in range(200)}
        with self.assertRaises(TensorSecurityError):
            validate_parameter_structure(large_params, max_params=100)
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality"""
        # Should return a float or 0.0 if psutil not available
        memory_usage = monitor_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertGreaterEqual(memory_usage, 0.0)
    
    def test_security_validation_edge_cases(self):
        """Test edge cases in security validation"""
        # Test empty parameter dict
        validate_parameter_structure({})
        
        # Test single parameter
        single_param = {'weight': [1.0, 2.0, 3.0]}
        validate_parameter_structure(single_param)
        
        # Test nested parameters
        nested_params = {
            'layer1': {
                'sublayer': {'weight': [1.0, 2.0]}
            }
        }
        validate_parameter_structure(nested_params)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    print("Running minimal tensor security tests...")
    unittest.main(verbosity=2)