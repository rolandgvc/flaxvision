from typing import Sequence

from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np


class DeepLabHead(nn.Module):
  num_classes: int

  @nn.compact
  def __call__(self, inputs, train: bool = False):
    x = ASPP([12, 24, 36], name='ASPP')(inputs)
    x = nn.Conv(256, (3, 3), padding='SAME', use_bias=False, name="conv1")(x)
    x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
    x = nn.relu(x)
    x = nn.Conv(self.num_classes, (1, 1), padding='VALID', use_bias=True, name="conv2")(x)
    return x


class ASPPConv(nn.Module):
  channels: int
  dilation: int

  @nn.compact
  def __call__(self, inputs, train: bool = False):
    _d = max(1, self.dilation)
    x = jnp.pad(inputs, [(0, 0), (_d, _d), (_d, _d), (0, 0)], 'constant', (0, 0))
    x = nn.Conv(self.channels, (3, 3), padding='VALID', kernel_dilation=(_d, _d), use_bias=False, name='conv1')(x)
    x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
    x = nn.relu(x)
    return x


class ASPPPooling(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, inputs, train: bool = False):
    in_shape = np.shape(inputs)[1:-1]
    x = nn.avg_pool(inputs, in_shape)
    x = nn.Conv(self.channels, (1, 1), padding='SAME', use_bias=False, name="conv1")(x)
    x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
    x = nn.relu(x)

    out_shape = (1, in_shape[0], in_shape[1], self.channels)
    x = jax.image.resize(x, shape=out_shape, method='bilinear')

    return x


class ASPP(nn.Module):
  atrous_rates: Sequence
  channels: int = 256

  @nn.compact
  def __call__(self, inputs, train: bool = False):
    res = []

    x = nn.Conv(self.channels, (1, 1), padding='VALID', use_bias=False, name="conv1")(inputs)
    x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
    res.append(nn.relu(x))

    for i, rate in enumerate(self.atrous_rates):
      res.append(ASPPConv(self.channels, rate, name=f'ASPPConv{i+1}')(inputs))

    res.append(ASPPPooling(self.channels, name='ASPPPooling')(inputs))
    x = jnp.concatenate(res, -1)  # 1280

    x = nn.Conv(self.channels, (1, 1), padding='VALID', use_bias=False, name="conv2")(x)
    x = nn.BatchNorm(use_running_average=not train, name="bn2")(x)
    x = nn.relu(x)
    x = nn.Dropout(0.5)(x, deterministic=not train)

    return x


def deeplabv3_keys(keys):
  layerblock = None
  layer_idx = None
  block_idx = None
  aspp_block = None

  if len(keys) == 3:  # first layer and classifier final block
    baseblock, layer, param = keys
  elif len(keys) == 5:  # block layer and classifier project block
    baseblock, layerblock, block_idx, layer, param = keys
  elif len(keys) == 6:  # downsample layer and classifier module block
    baseblock, layerblock, block_idx, layer, layer_idx, param = keys

  if 'aux' in baseblock or param == 'num_batches_tracked':
    return [None]

  if param == 'weight':
    param = 'scale' if 'bn' in layer else 'kernel'
  if 'running' in param:
    param = 'mean' if 'mean' in param else 'var'

  if baseblock == 'backbone':
    if layer_idx == '0':
      layer = 'downsample_conv'
    if layer_idx == '1':
      layer = 'downsample_bn'

    if 'bn' in layer and param == 'kernel':
      param = 'scale'

    if layerblock:
      return [baseblock, layerblock, f'block{int(block_idx)+1}', layer, param]

    return [baseblock, layer, param]

  elif baseblock == 'classifier':
    if layer_idx is not None:
      if block_idx == 'convs' and int(layer) < 4 and int(layer) > 0:
        aspp_block = f'ASPPConv{layer}'
      elif layer == '4':
        aspp_block = 'ASPPPooling'

    # ASPP first block
    if block_idx == 'convs' and layer == '0':
      layer = 'conv1' if layer_idx == '0' else 'bn1'

    # ASPPConv and ASPPPooling
    if block_idx == 'convs':
      if layer == '4':
        layer = 'conv1' if layer_idx == '1' else 'bn1'
      else:
        layer = 'conv1' if layer_idx == '0' else 'bn1'

    # ASPP final block
    if block_idx == 'project':
      if layer == '0':
        layer = 'conv2'
      if layer == '1':
        layer = 'bn2'

    layerblock = 'ASPP'

    # projection block
    if block_idx is None:
      if layer == '1':
        layer = 'conv1'
      if layer == '2':
        layer = 'bn1'
      if layer == '4':
        layer = 'conv2'
      layerblock = None

    if 'bn' in layer and param == 'kernel':
      param = 'scale'

    if layerblock is None:
      return [baseblock, layer, param]

    if aspp_block is None:
      return [baseblock, layerblock, layer, param]

    return [baseblock, layerblock, aspp_block, layer, param]
