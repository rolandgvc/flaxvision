#!/usr/bin/env python3
"""
Direct test for tensor security functions without package imports
"""

import sys
import os
import unittest
import logging

# Mock the required dependencies
class MockNumpy:
    def isfinite(self, x):
        # Return True if all elements are finite
        if hasattr(x, 'data'):
            return MockArray([not (val == float('inf') or val == float('-inf') or val != val) for val in x.data])
        elif hasattr(x, '__iter__'):
            return MockArray([not (val == float('inf') or val == float('-inf') or val != val) for val in x])
        else:
            return MockArray([not (x == float('inf') or x == float('-inf') or x != x)])
    
    def issubdtype(self, dtype, number):
        return True
    
    def number(self):
        return int
    
    def array(self, x):
        return MockArray(x)
    
    def transpose(self, x, axes=None):
        return x
    
    @property
    def inf(self):
        return float('inf')
    
    @property
    def nan(self):
        return float('nan')

class MockArray:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        self.shape = (len(self.data),)
        self.ndim = 1
        self.dtype = 'float32'
        self.nbytes = len(self.data) * 4
    
    def all(self):
        return all(self.data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        self.data[key] = value

class MockTorch:
    def randn(self, *args):
        return MockTensor([1.0] * (args[0] if args else 1))
    
    def tensor(self, data):
        return MockTensor(data)

class MockTensor:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
        self.shape = (len(self.data),)
    
    def detach(self):
        return self
    
    def numpy(self):
        return MockArray(self.data)

# Mock the modules
sys.modules['numpy'] = MockNumpy()
sys.modules['torch'] = MockTorch()
sys.modules['jax.numpy'] = MockNumpy()
sys.modules['jax'] = type('', (), {'numpy': MockNumpy()})()
sys.modules['flax'] = type('', (), {'nn': None})()
sys.modules['psutil'] = None

# Now directly import and execute the utils code
utils_path = os.path.join(os.path.dirname(__file__), 'flaxvision', 'utils.py')
with open(utils_path, 'r') as f:
    utils_code = f.read()

# Execute the utils code in a custom namespace
utils_namespace = {}
exec(utils_code, utils_namespace)

# Extract the functions we need
validate_tensor_security = utils_namespace['validate_tensor_security']
validate_parameter_structure = utils_namespace['validate_parameter_structure']
TensorSecurityError = utils_namespace['TensorSecurityError']
monitor_memory_usage = utils_namespace['monitor_memory_usage']

class TestTensorSecurityDirect(unittest.TestCase):
    """Direct tests for tensor security validation"""
    
    def test_tensor_security_validation_basic(self):
        """Test basic tensor security validation"""
        # Test with mock tensor
        mock_tensor = MockTensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = validate_tensor_security(mock_tensor)
        self.assertTrue(result is not None)
    
    def test_tensor_security_with_nan(self):
        """Test tensor security with NaN values"""
        # Test with NaN
        mock_tensor = MockTensor([1.0, float('nan'), 3.0])
        with self.assertRaises(TensorSecurityError):
            validate_tensor_security(mock_tensor)
    
    def test_tensor_security_with_inf(self):
        """Test tensor security with infinite values"""
        # Test with infinity
        mock_tensor = MockTensor([1.0, float('inf'), 3.0])
        with self.assertRaises(TensorSecurityError):
            validate_tensor_security(mock_tensor)
    
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
    
    def test_oversized_tensor_detection(self):
        """Test oversized tensor detection"""
        # Create a mock tensor that's "too large"
        mock_tensor = MockTensor([1.0] * 1000)
        
        # Should raise error with very small size limit
        with self.assertRaises(TensorSecurityError):
            validate_tensor_security(mock_tensor, max_size_gb=0.000001)
    
    def test_parameter_key_validation(self):
        """Test parameter key validation"""
        # Test with non-string key
        with self.assertRaises(TensorSecurityError):
            validate_parameter_structure({123: [1.0, 2.0]})
        
        # Test with very long key
        long_key = 'a' * 250
        with self.assertRaises(TensorSecurityError):
            validate_parameter_structure({long_key: [1.0, 2.0]})
    
    def test_tensor_dimension_validation(self):
        """Test tensor dimension validation"""
        # Mock a tensor with too many dimensions
        mock_tensor = MockTensor([1.0])
        
        # We need to mock the numpy conversion to return high-dimensional tensor
        original_numpy = mock_tensor.numpy
        def mock_numpy_high_dim():
            result = original_numpy()
            result.ndim = 7  # More than 6 dimensions
            return result
        mock_tensor.numpy = mock_numpy_high_dim
        
        with self.assertRaises(TensorSecurityError):
            validate_tensor_security(mock_tensor)

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    print("Running direct tensor security tests...")
    unittest.main(verbosity=2)