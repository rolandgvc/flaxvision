import unittest
import numpy as np
import jax.numpy as jnp
from jax import random
from unittest.mock import patch, Mock
import warnings

import flaxvision.utils as utils
import flaxvision.models as models


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error handling and resilience features."""

    def setUp(self):
        self.rng = random.PRNGKey(0)
        self.valid_input = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
        self.inception_input = jnp.ones((1, 299, 299, 3), dtype=jnp.float32)

    def test_load_torch_params_network_failure(self):
        """Test network failure handling with retry logic."""
        # Mock network failure
        with patch('torch.hub.load_state_dict_from_url') as mock_load:
            mock_load.side_effect = ConnectionError("Network error")
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = utils.load_torch_params("http://example.com/model.pth")
                
                # Should return None on failure
                self.assertIsNone(result)
                # Should have warning about network failure
                self.assertTrue(any("Network error" in str(warning.message) for warning in w))

    def test_load_torch_params_retry_success(self):
        """Test successful retry after initial failure."""
        mock_params = {'weight': np.ones((64, 3, 7, 7))}
        
        with patch('torch.hub.load_state_dict_from_url') as mock_load:
            # First call fails, second succeeds
            mock_load.side_effect = [ConnectionError("Network error"), mock_params]
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = utils.load_torch_params("http://example.com/model.pth")
                
                # Should return params on retry success
                self.assertEqual(result, mock_params)
                # Should have warning about retry
                self.assertTrue(any("Network error" in str(warning.message) for warning in w))

    def test_torch_to_linen_none_params(self):
        """Test torch_to_linen with None parameters."""
        with self.assertRaises(ValueError) as context:
            utils.torch_to_linen(None, lambda x: x)
        self.assertIn("torch_params cannot be None", str(context.exception))

    def test_torch_to_linen_invalid_type(self):
        """Test torch_to_linen with invalid parameter type."""
        with self.assertRaises(TypeError) as context:
            utils.torch_to_linen("not_a_dict", lambda x: x)
        self.assertIn("torch_params must be a dictionary", str(context.exception))

    def test_torch_to_linen_empty_params(self):
        """Test torch_to_linen with empty parameters."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = utils.torch_to_linen({}, lambda x: x)
            
            # Should return empty structure
            self.assertEqual(result, {'params': {}, 'batch_stats': {}})
            # Should have warning about empty params
            self.assertTrue(any("empty" in str(warning.message) for warning in w))

    def test_torch_to_linen_invalid_tensor(self):
        """Test torch_to_linen with invalid tensor."""
        mock_tensor = Mock()
        mock_tensor.detach.side_effect = AttributeError("Not a tensor")
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = utils.torch_to_linen(
                {'invalid_param': mock_tensor}, 
                lambda x: ['params', 'test']
            )
            
            # Should skip invalid parameter
            self.assertEqual(result['params'], {})
            # Should have warning about parameter error
            self.assertTrue(any("Error converting parameter" in str(warning.message) for warning in w))

    def test_resnet_input_validation_wrong_type(self):
        """Test ResNet input validation with wrong input type."""
        model, _ = models.resnet18(self.rng, pretrained=False)
        
        with self.assertRaises(TypeError) as context:
            model(np.ones((1, 224, 224, 3)))  # numpy array instead of JAX array
        self.assertIn("inputs must be a JAX array", str(context.exception))

    def test_resnet_input_validation_wrong_dimensions(self):
        """Test ResNet input validation with wrong dimensions."""
        model, _ = models.resnet18(self.rng, pretrained=False)
        
        with self.assertRaises(ValueError) as context:
            model(jnp.ones((1, 224, 224)))  # 3D instead of 4D
        self.assertIn("inputs must be 4-dimensional", str(context.exception))

    def test_resnet_input_validation_wrong_channels(self):
        """Test ResNet input validation with wrong number of channels."""
        model, _ = models.resnet18(self.rng, pretrained=False)
        
        with self.assertRaises(ValueError) as context:
            model(jnp.ones((1, 224, 224, 1)))  # 1 channel instead of 3
        self.assertIn("inputs must have 3 channels", str(context.exception))

    def test_resnet_input_validation_too_small(self):
        """Test ResNet input validation with too small dimensions."""
        model, _ = models.resnet18(self.rng, pretrained=False)
        
        with self.assertRaises(ValueError) as context:
            model(jnp.ones((1, 16, 16, 3)))  # 16x16 instead of minimum 32x32
        self.assertIn("inputs spatial dimensions must be at least 32x32", str(context.exception))

    def test_resnet_input_validation_nan_values(self):
        """Test ResNet input validation with NaN values."""
        model, _ = models.resnet18(self.rng, pretrained=False)
        
        invalid_input = jnp.ones((1, 224, 224, 3))
        invalid_input = invalid_input.at[0, 0, 0, 0].set(jnp.nan)
        
        with self.assertRaises(ValueError) as context:
            model(invalid_input)
        self.assertIn("inputs contains NaN values", str(context.exception))

    def test_resnet_input_validation_inf_values(self):
        """Test ResNet input validation with infinity values."""
        model, _ = models.resnet18(self.rng, pretrained=False)
        
        invalid_input = jnp.ones((1, 224, 224, 3))
        invalid_input = invalid_input.at[0, 0, 0, 0].set(jnp.inf)
        
        with self.assertRaises(ValueError) as context:
            model(invalid_input)
        self.assertIn("inputs contains infinity values", str(context.exception))

    def test_vgg_input_validation(self):
        """Test VGG input validation."""
        model, _ = models.vgg11(self.rng, pretrained=False)
        
        # Test wrong type
        with self.assertRaises(TypeError):
            model(np.ones((1, 224, 224, 3)))
        
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 224, 224)))
        
        # Test wrong channels
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 224, 224, 1)))

    def test_densenet_input_validation(self):
        """Test DenseNet input validation."""
        model, _ = models.densenet121(self.rng, pretrained=False)
        
        # Test wrong type
        with self.assertRaises(TypeError):
            model(np.ones((1, 224, 224, 3)))
        
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 224, 224)))
        
        # Test wrong channels
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 224, 224, 1)))

    def test_inception_input_validation(self):
        """Test Inception input validation."""
        model, _ = models.inception_v3(self.rng, pretrained=False)
        
        # Test wrong type
        with self.assertRaises(TypeError):
            model(np.ones((1, 299, 299, 3)))
        
        # Test wrong dimensions
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 299, 299)))
        
        # Test wrong channels
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 299, 299, 1)))
        
        # Test too small dimensions (Inception needs at least 75x75)
        with self.assertRaises(ValueError):
            model(jnp.ones((1, 50, 50, 3)))

    def test_model_factory_network_failure_fallback(self):
        """Test model factory functions handle network failures gracefully."""
        with patch('flaxvision.utils.load_torch_params') as mock_load:
            mock_load.return_value = None  # Simulate network failure
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Should not raise exception, should fallback to random initialization
                model, params = models.resnet18(self.rng, pretrained=True)
                
                # Should be able to run inference
                output = model.apply(params, self.valid_input)
                self.assertEqual(output.shape, (1, 1000))  # Should have correct output shape

    def test_model_factory_parameter_conversion_failure(self):
        """Test model factory functions handle parameter conversion failures."""
        # Mock successful download but failed conversion
        mock_params = {'invalid_key': 'invalid_value'}
        
        with patch('flaxvision.utils.load_torch_params') as mock_load:
            mock_load.return_value = mock_params
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Should not raise exception, should fallback to random initialization
                model, params = models.resnet18(self.rng, pretrained=True)
                
                # Should be able to run inference
                output = model.apply(params, self.valid_input)
                self.assertEqual(output.shape, (1, 1000))

    def test_segmentation_model_network_failure_fallback(self):
        """Test segmentation model factory functions handle network failures."""
        with patch('flaxvision.utils.load_torch_params') as mock_load:
            mock_load.return_value = None  # Simulate network failure
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Should not raise exception, should fallback to random initialization
                model, params = models.fcn_resnet50(self.rng, pretrained=True)
                
                # Should be able to run inference
                output = model.apply(params, self.valid_input)
                self.assertEqual(output.shape, (224, 224, 21))  # Should have correct output shape

    def test_valid_inputs_still_work(self):
        """Test that valid inputs still work correctly after adding validation."""
        # Test ResNet
        model, params = models.resnet18(self.rng, pretrained=False)
        output = model.apply(params, self.valid_input)
        self.assertEqual(output.shape, (1, 1000))
        
        # Test VGG
        model, params = models.vgg11(self.rng, pretrained=False)
        output = model.apply(params, self.valid_input)
        self.assertEqual(output.shape, (1, 1000))
        
        # Test DenseNet
        model, params = models.densenet121(self.rng, pretrained=False)
        output = model.apply(params, self.valid_input)
        self.assertEqual(output.shape, (1, 1000))
        
        # Test Inception
        model, params = models.inception_v3(self.rng, pretrained=False)
        output = model.apply(params, self.inception_input)
        self.assertEqual(output.shape, (1, 1000))


if __name__ == '__main__':
    unittest.main()