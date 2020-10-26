from typing import Any, Sequence
import functools
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from .. import utils

model_urls = {
  'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
  'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
  'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
  'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
  'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
  'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
  'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
  'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class Classifier(nn.Module):
  num_classes: int
  train: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    x = nn.Dense(4096, dtype=self.dtype)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=0.5)(x, deterministic=not self.train)
    x = nn.Dense(4096, dtype=self.dtype)(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=0.5)(x, deterministic=not self.train)
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    return x


class Features(nn.Module):
  cfg: Sequence[Any]
  batch_norm: bool = False
  train: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    for v in self.cfg:
      if v == 'M':
        x = nn.max_pool(x, (2, 2), (2, 2))
      else:
        x = nn.Conv(v, (3, 3), padding='SAME', dtype=self.dtype)(x)
        if self.batch_norm:
          x = nn.BatchNorm(use_running_average=not self.train, momentum=0.1, dtype=self.dtype)(x)
        x = nn.relu(x)
    return x


class VGG(nn.Module):
  rng: Any
  cfg: Sequence[Any]
  num_classes: int = 1000
  batch_norm: bool = False
  train: bool = False
  dtype: Any = jnp.float32

  def setup(self):
    self.features = Features(self.cfg, self.batch_norm, self.train, self.dtype)
    self.classifier = Classifier(self.num_classes, self.train)

  def __call__(self, inputs):
    x = self.features(inputs)
    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = self.classifier(x)
    return x


def _torch_to_flax(torch_params, cfg, batch_norm=False):
  """Convert PyTorch parameters to nested dictionaries."""
  flax_params = {'params': {'features': {}, 'classifier': {}},
                 'batch_stats': {'features': {}}}
  conv_idx = 0
  bn_idx = 0

  tensor_iter = iter(torch_params.items())
  def next_tensor():
    _, tensor = next(tensor_iter)
    return tensor.detach().numpy()

  for layer_cfg in cfg:
    if isinstance(layer_cfg, int):
      flax_params['params']['features'][f'Conv_{conv_idx}'] = {
        'kernel': np.transpose(next_tensor(), (2, 3, 1, 0)),
        'bias': next_tensor(),
      }
      conv_idx += 1

      if batch_norm:
        flax_params['params']['features'][f'BatchNorm_{bn_idx}'] = {
          'scale': next_tensor(),
          'bias': next_tensor(),
        }
        flax_params['batch_stats']['features'][f'BatchNorm_{bn_idx}'] = {
          'mean': next_tensor(),
          'var': next_tensor(),
        }
        bn_idx += 1

  for idx in range(3):
    flax_params['params']['classifier'][f'Dense_{idx}'] = {
      'kernel': np.transpose(next_tensor()),
      'bias': next_tensor(),
    }

  return FrozenDict(flax_params)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def _vgg(arch, cfg, rng, batch_norm, pretrained, **kwargs):
  vgg = functools.partial(VGG, rng=rng, cfg=cfgs[cfg], batch_norm=batch_norm, **kwargs)

  if pretrained:
    torch_params = utils.load_torch_params(model_urls[arch])
    flax_params = FrozenDict(_torch_to_flax(torch_params, cfgs[cfg], batch_norm))
  else:
    init_batch = jnp.ones((1, 224, 224, 3), jnp.float32)
    flax_params = VGG(rng=rng, cfg=cfgs[cfg], batch_norm=batch_norm, **kwargs).init(rng, init_batch)

  return vgg, flax_params


def vgg11(rng, pretrained=True, **kwargs):
  return _vgg('vgg11', 'A', rng, False, pretrained, **kwargs)


def vgg11_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg11_bn', 'A', rng, True, pretrained, **kwargs)


def vgg13(rng, pretrained=True, **kwargs):
  return _vgg('vgg13', 'B', rng, False, pretrained, **kwargs)


def vgg13_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg13_bn', 'B', rng, True, pretrained, **kwargs)


def vgg16(rng, pretrained=True, **kwargs):
  return _vgg('vgg16', 'D', rng, False, pretrained, **kwargs)


def vgg16_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg16_bn', 'D', rng, True, pretrained, **kwargs)


def vgg19(rng, pretrained=True, **kwargs):
  return _vgg('vgg19', 'E', rng, False, pretrained, **kwargs)


def vgg19_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg19_bn', 'E', rng, True, pretrained, **kwargs)
