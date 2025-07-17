import torch
import numpy as np
import jax.numpy as jnp
from flax import nn
import logging
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class TensorSecurityError(Exception):
    """Exception raised for tensor security violations"""
    pass


def validate_tensor_security(tensor, max_size_gb=1.0):
    """
    Validate tensor for security issues including:
    - Finite values (no NaN/Inf)
    - Memory size limits
    - Valid data types
    """
    if tensor is None:
        raise TensorSecurityError("Tensor cannot be None")
    
    # Convert to numpy if it's a torch tensor
    if hasattr(tensor, 'detach'):
        numpy_tensor = tensor.detach().numpy()
    else:
        numpy_tensor = tensor
    
    # Check for finite values
    if not np.isfinite(numpy_tensor).all():
        raise TensorSecurityError("Tensor contains non-finite values (NaN or Inf)")
    
    # Check tensor size limit (prevent memory exhaustion)
    tensor_size_bytes = numpy_tensor.nbytes
    max_size_bytes = max_size_gb * 1024 * 1024 * 1024
    
    if tensor_size_bytes > max_size_bytes:
        raise TensorSecurityError(f"Tensor size ({tensor_size_bytes} bytes) exceeds maximum allowed size ({max_size_bytes} bytes)")
    
    # Check for reasonable tensor dimensions
    if numpy_tensor.ndim > 6:  # More than 6 dimensions is suspicious
        raise TensorSecurityError(f"Tensor has too many dimensions ({numpy_tensor.ndim})")
    
    # Check for valid data types
    if not np.issubdtype(numpy_tensor.dtype, np.number):
        raise TensorSecurityError(f"Tensor has invalid data type: {numpy_tensor.dtype}")
    
    return numpy_tensor


def validate_parameter_structure(params_dict, expected_keys=None, max_params=10000):
    """
    Validate parameter dictionary structure:
    - Check parameter count limits
    - Validate key structure
    - Ensure reasonable parameter sizes
    """
    if not isinstance(params_dict, dict):
        raise TensorSecurityError("Parameters must be a dictionary")
    
    # Check parameter count
    total_params = len(params_dict)
    if total_params > max_params:
        raise TensorSecurityError(f"Too many parameters ({total_params}), maximum allowed: {max_params}")
    
    # Validate each parameter
    for key, value in params_dict.items():
        if not isinstance(key, str):
            raise TensorSecurityError(f"Parameter key must be string, got: {type(key)}")
        
        if len(key) > 200:  # Prevent extremely long keys
            raise TensorSecurityError(f"Parameter key too long: {len(key)} characters")
        
        # Recursively validate nested dictionaries
        if isinstance(value, dict):
            validate_parameter_structure(value, expected_keys, max_params)


def monitor_memory_usage():
    """Monitor current memory usage and warn if approaching limits"""
    if not PSUTIL_AVAILABLE:
        logging.warning("psutil not available, memory monitoring disabled")
        return 0.0
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # Warning threshold at 2GB
        if memory_mb > 2048:
            logging.warning(f"High memory usage detected: {memory_mb:.2f} MB")
        
        return memory_mb
    except Exception as e:
        logging.warning(f"Failed to monitor memory usage: {e}")
        return 0.0


def load_torch_params(url):
    return torch.hub.load_state_dict_from_url(url)


def torch_to_flax(torch_params, get_flax_keys):
  """Convert PyTorch parameters to nested dictionaries with security validation"""
  
  # Monitor memory usage at start
  initial_memory = monitor_memory_usage()
  
  # Validate input parameter structure
  validate_parameter_structure(torch_params)
  
  def add_to_params(params_dict, nested_keys, param, is_conv=False):
    if len(nested_keys) == 1:
      key, = nested_keys
      params_dict[key] = np.transpose(param, (2, 3, 1, 0)) if is_conv else np.transpose(param)
    else:
      assert len(nested_keys) > 1
      first_key = nested_keys[0]
      if first_key not in params_dict:
        params_dict[first_key] = {}
      add_to_params(params_dict[first_key], nested_keys[1:], param, ('conv' in first_key and \
                                                                     nested_keys[-1] != 'bias'))

  def add_to_state(state_dict, keys, param):
    key_str = ''
    for k in keys[:-1]:
      key_str += f"/{k}"
    if key_str not in state_dict:
      state_dict[key_str] = {}
    state_dict[key_str][keys[-1]] = param

  flax_params, flax_state = {}, {}
  for key, tensor in torch_params.items():
    # Validate tensor security before processing
    try:
      validated_tensor = validate_tensor_security(tensor)
    except TensorSecurityError as e:
      logging.error(f"Tensor security validation failed for key '{key}': {e}")
      raise
    
    flax_keys = get_flax_keys(key.split('.'))
    if flax_keys[-1] is None:
      continue
    if flax_keys[-1] == 'mean' or flax_keys[-1] == 'var':
      add_to_state(flax_state, flax_keys, validated_tensor)
    else:
      add_to_params(flax_params, flax_keys, validated_tensor)
    
    # Monitor memory usage during processing
    current_memory = monitor_memory_usage()
    if current_memory > initial_memory + 1000:  # 1GB increase
      logging.warning(f"Memory usage increased significantly during parameter conversion: {current_memory - initial_memory:.2f} MB")

  # Validate final parameter structures
  validate_parameter_structure(flax_params)
  validate_parameter_structure(flax_state)

  return flax_params, flax_state


def torch_to_linen(torch_params, get_flax_keys):
  """Convert PyTorch parameters to Linen nested dictionaries with security validation"""
  
  # Monitor memory usage at start
  initial_memory = monitor_memory_usage()
  
  # Validate input parameter structure
  validate_parameter_structure(torch_params)

  def add_to_params(params_dict, nested_keys, param, is_conv=False):
    if len(nested_keys) == 1:
      key, = nested_keys
      params_dict[key] = np.transpose(param, (2, 3, 1, 0)) if is_conv else np.transpose(param)
    else:
      assert len(nested_keys) > 1
      first_key = nested_keys[0]
      if first_key not in params_dict:
        params_dict[first_key] = {}
      add_to_params(params_dict[first_key], nested_keys[1:], param, ('conv' in first_key and \
                                                                     nested_keys[-1] != 'bias'))

  flax_params = {'params': {}, 'batch_stats': {}}
  for key, tensor in torch_params.items():
    # Validate tensor security before processing
    try:
      validated_tensor = validate_tensor_security(tensor)
    except TensorSecurityError as e:
      logging.error(f"Tensor security validation failed for key '{key}': {e}")
      raise
    
    flax_keys = get_flax_keys(key.split('.'))
    if flax_keys[-1] is not None:
      if flax_keys[-1] in ('mean', 'var'):
        add_to_params(flax_params['batch_stats'], flax_keys, validated_tensor)
      else:
        add_to_params(flax_params['params'], flax_keys, validated_tensor)
    
    # Monitor memory usage during processing
    current_memory = monitor_memory_usage()
    if current_memory > initial_memory + 1000:  # 1GB increase
      logging.warning(f"Memory usage increased significantly during parameter conversion: {current_memory - initial_memory:.2f} MB")

  # Validate final parameter structures
  validate_parameter_structure(flax_params['params'])
  validate_parameter_structure(flax_params['batch_stats'])

  return flax_params
