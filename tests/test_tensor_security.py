"""
Comprehensive test suite for tensor security validation in flaxvision.utils
Tests malicious tensor inputs and parameter conversion security.
"""

import pytest
import numpy as np
import torch
import sys
import os

# Add the parent directory to the path to import flaxvision
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flaxvision.utils import (
    validate_tensor_security,
    validate_parameter_structure,
    torch_to_flax,
    torch_to_linen,
    TensorSecurityError,
    monitor_memory_usage
)


class TestTensorSecurityValidation:
    """Test tensor security validation functions"""
    
    def test_valid_tensor_passes_validation(self):
        """Test that valid tensors pass security validation"""
        # Test with torch tensor
        torch_tensor = torch.randn(10, 10)
        validated = validate_tensor_security(torch_tensor)
        assert validated.shape == (10, 10)
        
        # Test with numpy tensor
        numpy_tensor = np.random.randn(5, 5)
        validated = validate_tensor_security(numpy_tensor)
        assert validated.shape == (5, 5)
    
    def test_nan_tensor_raises_error(self):
        """Test that tensors with NaN values raise security error"""
        # Torch tensor with NaN
        torch_tensor = torch.randn(10, 10)
        torch_tensor[0, 0] = float('nan')
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            validate_tensor_security(torch_tensor)
        
        # Numpy tensor with NaN
        numpy_tensor = np.random.randn(5, 5)
        numpy_tensor[0, 0] = np.nan
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            validate_tensor_security(numpy_tensor)
    
    def test_inf_tensor_raises_error(self):
        """Test that tensors with infinite values raise security error"""
        # Torch tensor with Inf
        torch_tensor = torch.randn(10, 10)
        torch_tensor[0, 0] = float('inf')
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            validate_tensor_security(torch_tensor)
        
        # Numpy tensor with Inf
        numpy_tensor = np.random.randn(5, 5)
        numpy_tensor[0, 0] = np.inf
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            validate_tensor_security(numpy_tensor)
    
    def test_oversized_tensor_raises_error(self):
        """Test that oversized tensors raise security error"""
        # Create a tensor that's too large (>1GB default limit)
        # We'll use a smaller limit for testing
        small_tensor = torch.randn(100, 100)
        
        with pytest.raises(TensorSecurityError, match="exceeds maximum allowed size"):
            validate_tensor_security(small_tensor, max_size_gb=0.0001)  # Very small limit
    
    def test_too_many_dimensions_raises_error(self):
        """Test that tensors with too many dimensions raise security error"""
        # Create a 7-dimensional tensor (limit is 6)
        shape = [2] * 7
        tensor = torch.randn(*shape)
        
        with pytest.raises(TensorSecurityError, match="too many dimensions"):
            validate_tensor_security(tensor)
    
    def test_invalid_data_type_raises_error(self):
        """Test that tensors with invalid data types raise security error"""
        # Create a tensor with object dtype
        numpy_tensor = np.array([{'a': 1}, {'b': 2}], dtype=object)
        
        with pytest.raises(TensorSecurityError, match="invalid data type"):
            validate_tensor_security(numpy_tensor)
    
    def test_none_tensor_raises_error(self):
        """Test that None tensor raises security error"""
        with pytest.raises(TensorSecurityError, match="Tensor cannot be None"):
            validate_tensor_security(None)


class TestParameterStructureValidation:
    """Test parameter structure validation functions"""
    
    def test_valid_parameter_structure_passes(self):
        """Test that valid parameter structures pass validation"""
        params = {
            'layer1': {'weight': np.random.randn(10, 10), 'bias': np.random.randn(10)},
            'layer2': {'weight': np.random.randn(5, 10), 'bias': np.random.randn(5)}
        }
        
        # Should not raise exception
        validate_parameter_structure(params)
    
    def test_non_dict_parameter_raises_error(self):
        """Test that non-dictionary parameters raise error"""
        params = "not a dictionary"
        
        with pytest.raises(TensorSecurityError, match="Parameters must be a dictionary"):
            validate_parameter_structure(params)
    
    def test_too_many_parameters_raises_error(self):
        """Test that too many parameters raise error"""
        params = {f'param_{i}': np.random.randn(2, 2) for i in range(100)}
        
        with pytest.raises(TensorSecurityError, match="Too many parameters"):
            validate_parameter_structure(params, max_params=50)
    
    def test_non_string_key_raises_error(self):
        """Test that non-string keys raise error"""
        params = {123: np.random.randn(10, 10)}
        
        with pytest.raises(TensorSecurityError, match="Parameter key must be string"):
            validate_parameter_structure(params)
    
    def test_long_key_raises_error(self):
        """Test that extremely long keys raise error"""
        long_key = 'a' * 250  # Longer than 200 character limit
        params = {long_key: np.random.randn(10, 10)}
        
        with pytest.raises(TensorSecurityError, match="Parameter key too long"):
            validate_parameter_structure(params)
    
    def test_nested_parameter_validation(self):
        """Test that nested parameter structures are validated"""
        params = {
            'layer1': {
                'nested': {
                    123: np.random.randn(10, 10)  # Invalid key type in nested structure
                }
            }
        }
        
        with pytest.raises(TensorSecurityError, match="Parameter key must be string"):
            validate_parameter_structure(params)


class TestSecureTensorConversion:
    """Test secure tensor conversion functions"""
    
    def get_dummy_flax_keys(self, torch_keys):
        """Dummy function to convert torch keys to flax keys"""
        return torch_keys  # Simple passthrough for testing
    
    def test_torch_to_flax_with_valid_tensors(self):
        """Test torch_to_flax with valid tensors"""
        torch_params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(5, 10)
        }
        
        flax_params, flax_state = torch_to_flax(torch_params, self.get_dummy_flax_keys)
        
        assert isinstance(flax_params, dict)
        assert isinstance(flax_state, dict)
    
    def test_torch_to_linen_with_valid_tensors(self):
        """Test torch_to_linen with valid tensors"""
        torch_params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(5, 10)
        }
        
        linen_params = torch_to_linen(torch_params, self.get_dummy_flax_keys)
        
        assert isinstance(linen_params, dict)
        assert 'params' in linen_params
        assert 'batch_stats' in linen_params
    
    def test_torch_to_flax_with_malicious_tensors(self):
        """Test torch_to_flax rejects malicious tensors"""
        # Test with NaN tensor
        torch_params = {
            'layer1.weight': torch.tensor([[float('nan'), 1.0], [2.0, 3.0]]),
        }
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            torch_to_flax(torch_params, self.get_dummy_flax_keys)
        
        # Test with infinite tensor
        torch_params = {
            'layer1.weight': torch.tensor([[float('inf'), 1.0], [2.0, 3.0]]),
        }
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            torch_to_flax(torch_params, self.get_dummy_flax_keys)
    
    def test_torch_to_linen_with_malicious_tensors(self):
        """Test torch_to_linen rejects malicious tensors"""
        # Test with NaN tensor
        torch_params = {
            'layer1.weight': torch.tensor([[float('nan'), 1.0], [2.0, 3.0]]),
        }
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            torch_to_linen(torch_params, self.get_dummy_flax_keys)
        
        # Test with infinite tensor
        torch_params = {
            'layer1.weight': torch.tensor([[float('inf'), 1.0], [2.0, 3.0]]),
        }
        
        with pytest.raises(TensorSecurityError, match="non-finite values"):
            torch_to_linen(torch_params, self.get_dummy_flax_keys)
    
    def test_torch_to_flax_with_oversized_tensors(self):
        """Test torch_to_flax rejects oversized tensors"""
        # Create a reasonably sized tensor for testing
        torch_params = {
            'layer1.weight': torch.randn(100, 100),
        }
        
        # Mock the validation to use a very small size limit
        original_validate = validate_tensor_security
        
        def mock_validate(tensor, max_size_gb=0.0001):  # Very small limit
            return original_validate(tensor, max_size_gb)
        
        import flaxvision.utils
        flaxvision.utils.validate_tensor_security = mock_validate
        
        try:
            with pytest.raises(TensorSecurityError, match="exceeds maximum allowed size"):
                torch_to_flax(torch_params, self.get_dummy_flax_keys)
        finally:
            # Restore original function
            flaxvision.utils.validate_tensor_security = original_validate
    
    def test_torch_to_linen_with_batch_stats(self):
        """Test torch_to_linen properly handles batch statistics"""
        torch_params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'bn1.mean': torch.randn(10),
            'bn1.var': torch.randn(10)
        }
        
        def get_flax_keys_with_bn(torch_keys):
            if 'mean' in torch_keys or 'var' in torch_keys:
                return torch_keys  # Return as-is for batch norm stats
            return torch_keys
        
        linen_params = torch_to_linen(torch_params, get_flax_keys_with_bn)
        
        assert isinstance(linen_params, dict)
        assert 'params' in linen_params
        assert 'batch_stats' in linen_params


class TestMemoryMonitoring:
    """Test memory monitoring functionality"""
    
    def test_memory_monitoring_returns_float(self):
        """Test that memory monitoring returns a float value"""
        memory_usage = monitor_memory_usage()
        assert isinstance(memory_usage, float)
        assert memory_usage > 0
    
    def test_memory_monitoring_detects_usage(self):
        """Test that memory monitoring can detect memory usage"""
        # Allocate some memory
        large_array = np.random.randn(1000, 1000)
        
        memory_usage = monitor_memory_usage()
        assert memory_usage > 0
        
        # Clean up
        del large_array


class TestEdgeCases:
    """Test edge cases and corner scenarios"""
    
    def test_empty_parameter_dict(self):
        """Test handling of empty parameter dictionaries"""
        empty_params = {}
        
        # Should not raise exception
        validate_parameter_structure(empty_params)
    
    def test_single_parameter(self):
        """Test handling of single parameter"""
        single_param = {'weight': torch.randn(10, 10)}
        
        def get_flax_keys(torch_keys):
            return torch_keys
        
        flax_params, flax_state = torch_to_flax(single_param, get_flax_keys)
        assert isinstance(flax_params, dict)
        assert isinstance(flax_state, dict)
    
    def test_parameter_with_none_flax_keys(self):
        """Test handling of parameters that map to None flax keys"""
        torch_params = {
            'ignored_param': torch.randn(10, 10),
            'valid_param': torch.randn(5, 5)
        }
        
        def get_flax_keys_with_none(torch_keys):
            if 'ignored' in torch_keys[0]:
                return [None]
            return torch_keys
        
        flax_params, flax_state = torch_to_flax(torch_params, get_flax_keys_with_none)
        
        # Should process only the valid parameter
        assert isinstance(flax_params, dict)
        assert isinstance(flax_state, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])