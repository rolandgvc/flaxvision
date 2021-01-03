from flax import linen as nn
import numpy as np


class FCNHead(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, inputs, train: bool = False):
    inter_channels = np.shape(inputs)[-1] // 4
    x = nn.Conv(inter_channels, (3, 3), padding='SAME', use_bias=False, name="conv1")(inputs)
    x = nn.BatchNorm(use_running_average=not train, name="bn1")(x)
    x = nn.relu(x)
    x = nn.Dropout(0.1)(x, deterministic=not train)
    x = nn.Conv(self.channels, (1, 1), padding='VALID', use_bias=True, name="conv2")(x)

    return x


def fcn_keys(keys):
  layerblock = None
  layer_idx = None

  if len(keys) == 3:  # first layer and classifier
    baseblock, layer, param = keys
  elif len(keys) == 5:  # block layer
    baseblock, layerblock, block_idx, layer, param = keys
  elif len(keys) == 6:  # downsample layer
    baseblock, layerblock, block_idx, layer, layer_idx, param = keys

  if 'aux' in baseblock or param == 'num_batches_tracked':
    return [None]

  if layer_idx == '0':
    layer = 'downsample_conv'
  if layer_idx == '1':
    layer = 'downsample_bn'

  if param == 'weight':
    param = 'scale' if 'bn' in layer else 'kernel'
  if 'running' in param:
    param = 'mean' if 'mean' in param else 'var'

  if baseblock == 'backbone':
    if layerblock:
      return [baseblock, layerblock, f'block{int(block_idx)+1}', layer, param]
    return [baseblock, layer, param]
  elif baseblock == 'classifier':
    layer = 'conv1' if layer == '0' else ('bn1' if layer == '1' else 'conv2')
    if 'bn' in layer and param == 'kernel':
      param = 'scale'
    return [baseblock, layer, param]
