import torch
import numpy as np
import jax.numpy as jnp
from flax import nn


def load_torch_params(url):
  return torch.hub.load_state_dict_from_url(url)


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
      add_to_params(params_dict[first_key], nested_keys[1:], param, 'conv' in first_key)

  def add_to_state(state_dict, keys, param):
    key_str = ''
    for k in keys[:-1]:
      key_str += f"/{k}"
    if key_str not in state_dict:
      state_dict[key_str] = {}
    state_dict[key_str][keys[-1]] = param

  flax_params, flax_state = {}, {}
  for key, tensor in torch_params.items():
    flax_keys = get_flax_keys(key.split('.'))
    if flax_keys[-1] == 'mean' or flax_keys[-1] == 'var':
      add_to_state(flax_state, flax_keys, tensor.detach().numpy())
    else:
      add_to_params(flax_params, flax_keys, tensor.detach().numpy())

  return flax_params, flax_state


def torch_to_linen(torch_params, get_flax_keys):
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
      add_to_params(params_dict[first_key], nested_keys[1:], param, 'conv' in first_key)

  flax_params = {'params':{}, 'batch_stats':{}}
  for key, tensor in torch_params.items():
    flax_keys = get_flax_keys(key.split('.'))
    if flax_keys[-1] in ('mean', 'var'):
      add_to_params(flax_params['batch_stats'], flax_keys, tensor.detach().numpy())
    else:
      add_to_params(flax_params['params'], flax_keys, tensor.detach().numpy())

  return flax_params
