from typing import Any, Sequence, Union

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
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, train=False):
    x = nn.Dense(4096, dtype=self.dtype)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=0.5)(x, deterministic=not train)
    x = nn.Dense(4096, dtype=self.dtype)(x)
    x = nn.relu(x)
    x = nn.Dropout(rate=0.5)(x, deterministic=not train)
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    return x


class Backbone(nn.Module):
  cfg: Union[Sequence[int], Sequence[str]]
  batch_norm: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train=False):
    for v in self.cfg:
      if v == 'M':
        x = nn.max_pool(x, (2, 2), (2, 2))
      else:
        x = nn.Conv(v, (3, 3), padding='SAME', dtype=self.dtype)(x)
        if self.batch_norm:
          x = nn.BatchNorm(use_running_average=not train, momentum=0.1, dtype=self.dtype)(x)
        x = nn.relu(x)
    return x


class VGG(nn.Module):
  cfg: Union[Sequence[int], Sequence[str]]
  num_classes: int = 1000
  batch_norm: bool = False
  dtype: Any = jnp.float32

  @staticmethod
  def make_backbone(self):
    return Backbone(self.cfg, self.batch_norm, self.dtype)

  def setup(self):
    self.backbone = VGG.make_backbone(self)
    self.classifier = Classifier(self.num_classes, self.dtype)

  def __call__(self, inputs, train=False):
    x = self.backbone(inputs, train)
    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = self.classifier(x, train)
    return x


def _torch_to_vgg(torch_params, cfg, batch_norm=False):
  """Convert PyTorch parameters to nested dictionaries."""
  flax_params = {'params': {'backbone': {}, 'classifier': {}}, 'batch_stats': {'backbone': {}}}
  conv_idx = 0
  bn_idx = 0

  tensor_iter = iter(torch_params.items())

  def next_tensor():
    _, tensor = next(tensor_iter)
    return tensor.detach().numpy()

  for layer_cfg in cfg:
    if isinstance(layer_cfg, int):
      flax_params['params']['backbone'][f'Conv_{conv_idx}'] = {
          'kernel': np.transpose(next_tensor(), (2, 3, 1, 0)),
          'bias': next_tensor(),
      }
      conv_idx += 1

      if batch_norm:
        flax_params['params']['backbone'][f'BatchNorm_{bn_idx}'] = {
            'scale': next_tensor(),
            'bias': next_tensor(),
        }
        flax_params['batch_stats']['backbone'][f'BatchNorm_{bn_idx}'] = {
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
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def _vgg(rng, arch, cfg, batch_norm, pretrained, **kwargs):
  vgg = VGG(cfg=cfgs[cfg], batch_norm=batch_norm, **kwargs)

  if pretrained:
    torch_params = utils.load_torch_params(model_urls[arch])
    flax_params = FrozenDict(_torch_to_vgg(torch_params, cfgs[cfg], batch_norm))
  else:
    init_batch = jnp.ones((1, 224, 224, 3), jnp.float32)
    flax_params = VGG(cfg=cfgs[cfg], batch_norm=batch_norm, **kwargs).init(rng, init_batch)

  return vgg, flax_params


def vgg11(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg11', 'A', False, pretrained, **kwargs)


def vgg11_bn(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg11_bn', 'A', True, pretrained, **kwargs)


def vgg13(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg13', 'B', False, pretrained, **kwargs)


def vgg13_bn(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg13_bn', 'B', True, pretrained, **kwargs)


def vgg16(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg16', 'D', False, pretrained, **kwargs)


def vgg16_bn(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg16_bn', 'D', True, pretrained, **kwargs)


def vgg19(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg19', 'E', False, pretrained, **kwargs)


def vgg19_bn(rng, pretrained=True, **kwargs):
  return _vgg(rng, 'vgg19_bn', 'E', True, pretrained, **kwargs)
