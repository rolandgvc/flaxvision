from typing import Optional, Sequence, Tuple, Any
from functools import partial
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from .. import utils

ModuleDef = Any

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


class DenseLayer(nn.Module):
  growth_rate: int
  bn_size: int
  drop_rate: int

  @nn.compact
  def __call__(self, x, train: bool = False):
    x = jnp.concatenate(x, 3)

    x = nn.BatchNorm(use_running_average=not train, name='norm1')(x)
    x = nn.relu(x)
    x = nn.Conv(self.bn_size * self.growth_rate, (1, 1), (1, 1), padding='VALID', use_bias=False, name='conv1')(x)
    x = nn.BatchNorm(use_running_average=not train, name='norm2')(x)
    x = nn.relu(x)
    x = nn.Conv(self.growth_rate, (3, 3), (1, 1), padding='SAME', use_bias=False, name='conv2')(x)

    if self.drop_rate:
      x = nn.Dropout(rate=self.drop_rate)(x, deterministic=not train)

    return x


class DenseBlock(nn.Module):
  num_layers: int
  bn_size: int
  growth_rate: int
  drop_rate: int

  @nn.compact
  def __call__(self, x, train: bool = False):
    backbone = [x]
    for i in range(self.num_layers):
      backbone.append(
          DenseLayer(self.growth_rate, self.bn_size, self.drop_rate, name=f'denselayer{i+1}')(backbone, train))
    return jnp.concatenate(backbone, 3)


class Transition(nn.Module):
  output_features: int

  @nn.compact
  def __call__(self, x, train: bool = False):
    x = nn.BatchNorm(use_running_average=not train, name='norm')(x)
    x = nn.relu(x)
    x = nn.Conv(self.output_features, (1, 1), (1, 1), padding='VALID', use_bias=False, name='conv')(x)
    x = nn.avg_pool(x, (2, 2), (2, 2))
    return x


class Backbone(nn.Module):
  growth_rate: int = 32
  block_config: Tuple[int, int, int, int] = (6, 12, 24, 16)
  num_init_features: int = 64
  bn_size: int = 4
  drop_rate: int = 0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x, train: bool = False):
    # initblock
    x = nn.Conv(self.num_init_features, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], use_bias=False, name='conv0')(x)
    x = nn.BatchNorm(use_running_average=not train, name='norm0')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), (2, 2), padding=[(1, 1), (1, 1)])

    # denseblocks
    num_features = self.num_init_features
    for i, num_layers in enumerate(self.block_config):
      x = DenseBlock(num_layers, self.bn_size, self.growth_rate, self.drop_rate, name=f'denseblock{i+1}')(x, train)
      num_features += num_layers * self.growth_rate

      if i != len(self.block_config) - 1:
        num_features = num_features // 2
        x = Transition(num_features, name=f'transition{i+1}')(x, train)

    # finalblock
    x = nn.BatchNorm(use_running_average=not train, name='norm5')(x)

    x = nn.relu(x)
    x = nn.avg_pool(x, (7, 7))

    return x


class DenseNet(nn.Module):
  growth_rate: int = 32
  block_config: Tuple[int, int, int, int] = (6, 12, 24, 16)
  num_init_features: int = 64
  bn_size: int = 4
  drop_rate: int = 0
  num_classes: int = 1000
  dtype: Any = jnp.float32

  @staticmethod
  def make_backbone(self):
    return Backbone(self.growth_rate, self.block_config, self.num_init_features, self.bn_size, self.drop_rate,
                    self.dtype)

  def setup(self):
    self.backbone = DenseNet.make_backbone(self)
    self.classifier = nn.Dense(self.num_classes, dtype=self.dtype)

  def __call__(self, x, train: bool = False):
    x = self.backbone(x, train)
    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = self.classifier(x)

    return x


def _get_flax_keys(keys):
  if keys[0] == 'features':
    keys[0] = 'backbone'
  if keys[-1] == 'weight':
    is_scale = 'norm' in keys[-2] if len(keys) < 6 else 'norm' in keys[-3]
    keys[-1] = 'scale' if is_scale else 'kernel'
  if 'running' in keys[-1]:
    keys[-1] = 'mean' if 'mean' in keys[-1] else 'var'
  if keys[-2] in ('1', '2'):  # if index separated from layer name, concatenate
    keys = keys[:3] + [keys[3] + keys[4]] + [keys[5]]
  return keys


def _densenet(rng, arch, growth_rate, block_config, num_init_features, pretrained, **kwargs):
  densenet = DenseNet(growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features, **kwargs)

  if pretrained:
    torch_params = utils.load_torch_params(model_urls[arch])
    flax_params = FrozenDict(utils.torch_to_linen(torch_params, _get_flax_keys))
  else:
    init_batch = jnp.ones((1, 224, 224, 3), jnp.float32)
    flax_params = DenseNet(
        growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features,
        **kwargs).init(rng, init_batch)

  return densenet, flax_params


def densenet121(rng, pretrained=True, **kwargs):
  return _densenet(rng, 'densenet121', 32, (6, 12, 24, 16), 64, pretrained, **kwargs)


def densenet161(rng, pretrained=True, **kwargs):
  return _densenet(rng, 'densenet161', 48, (6, 12, 36, 24), 96, pretrained, **kwargs)


def densenet169(rng, pretrained=True, **kwargs):
  return _densenet(rng, 'densenet169', 32, (6, 12, 32, 32), 64, pretrained, **kwargs)


def densenet201(rng, pretrained=True, **kwargs):
  return _densenet(rng, 'densenet201', 32, (6, 12, 48, 32), 64, pretrained, **kwargs)
