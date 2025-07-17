import torch
import numpy as np
import jax.numpy as jnp
from flax import nn
import time
import warnings
from urllib.error import URLError
from requests.exceptions import ConnectionError, Timeout, RequestException


def load_torch_params(url, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
  """Load PyTorch parameters from URL with retry logic and error handling.
  
  Args:
    url: URL to download parameters from
    max_retries: Maximum number of retry attempts
    initial_delay: Initial delay between retries (seconds)
    backoff_factor: Exponential backoff factor
    
  Returns:
    Loaded parameters dict or None if all retries failed
  """
  last_exception = None
  delay = initial_delay
  
  for attempt in range(max_retries + 1):
    try:
      return torch.hub.load_state_dict_from_url(url)
    except (ConnectionError, URLError, Timeout, RequestException) as e:
      last_exception = e
      if attempt < max_retries:
        warnings.warn(f"Network error on attempt {attempt + 1}/{max_retries + 1}: {e}. Retrying in {delay}s...")
        time.sleep(delay)
        delay *= backoff_factor
      else:
        warnings.warn(f"Failed to download parameters after {max_retries + 1} attempts. Last error: {e}. Using random initialization.")
        return None
    except Exception as e:
      warnings.warn(f"Unexpected error loading parameters: {e}. Using random initialization.")
      return None
  
  return None


def torch_to_flax(torch_params, get_flax_keys):
  """Convert PyTorch parameters to nested dictionaries"""

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
    if flax_keys[-1] is None:
      continue
    flax_keys = get_flax_keys(key.split('.'))
    if flax_keys[-1] == 'mean' or flax_keys[-1] == 'var':
      add_to_state(flax_state, flax_keys, tensor.detach().numpy())
    else:
      add_to_params(flax_params, flax_keys, tensor.detach().numpy())

  return flax_params, flax_state


def torch_to_linen(torch_params, get_flax_keys):
  """Convert PyTorch parameters to Linen nested dictionaries with comprehensive error handling.
  
  Args:
    torch_params: PyTorch parameters dictionary
    get_flax_keys: Function to convert PyTorch keys to Flax keys
    
  Returns:
    Flax parameters dictionary with proper error handling
  """
  if torch_params is None:
    raise ValueError("torch_params cannot be None")
  
  if not isinstance(torch_params, dict):
    raise TypeError(f"torch_params must be a dictionary, got {type(torch_params)}")
  
  if not torch_params:
    warnings.warn("torch_params is empty, returning empty parameter structure")
    return {'params': {}, 'batch_stats': {}}

  def add_to_params(params_dict, nested_keys, param, is_conv=False):
    try:
      if len(nested_keys) == 1:
        key, = nested_keys
        if not hasattr(param, 'detach'):
          raise ValueError(f"Parameter {key} is not a valid tensor")
        
        param_np = param.detach().numpy()
        
        # Validate tensor before transpose
        if not isinstance(param_np, np.ndarray):
          raise ValueError(f"Parameter {key} conversion to numpy failed")
        
        if np.isnan(param_np).any() or np.isinf(param_np).any():
          raise ValueError(f"Parameter {key} contains NaN or infinity values")
        
        if is_conv:
          if param_np.ndim != 4:
            raise ValueError(f"Conv parameter {key} must be 4D, got {param_np.ndim}D")
          params_dict[key] = np.transpose(param_np, (2, 3, 1, 0))
        else:
          if param_np.ndim < 1:
            raise ValueError(f"Parameter {key} must be at least 1D, got {param_np.ndim}D")
          params_dict[key] = np.transpose(param_np) if param_np.ndim > 1 else param_np
      else:
        assert len(nested_keys) > 1
        first_key = nested_keys[0]
        if first_key not in params_dict:
          params_dict[first_key] = {}
        add_to_params(params_dict[first_key], nested_keys[1:], param, ('conv' in first_key and \
                                                                       nested_keys[-1] != 'bias'))
    except Exception as e:
      warnings.warn(f"Error converting parameter {nested_keys}: {e}. Skipping parameter.")

  flax_params = {'params': {}, 'batch_stats': {}}
  
  for key, tensor in torch_params.items():
    try:
      flax_keys = get_flax_keys(key.split('.'))
      if flax_keys is None:
        warnings.warn(f"get_flax_keys returned None for key {key}. Skipping.")
        continue
      
      if len(flax_keys) == 0:
        warnings.warn(f"get_flax_keys returned empty list for key {key}. Skipping.")
        continue
        
      if flax_keys[-1] is not None:
        if flax_keys[-1] in ('mean', 'var'):
          add_to_params(flax_params['batch_stats'], flax_keys, tensor)
        else:
          add_to_params(flax_params['params'], flax_keys, tensor)
    except Exception as e:
      warnings.warn(f"Error processing parameter {key}: {e}. Skipping parameter.")
      continue

  return flax_params
