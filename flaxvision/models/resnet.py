from typing import Any, Sequence, Dict
import functools
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from .. import utils


model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
  'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
  'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
  'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
  'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


conv1x1 = functools.partial(nn.Conv, kernel_size=(1, 1), padding='VALID', use_bias=False)


def conv3x3(x, features, strides=(1, 1), groups=1, dilation=1, name='conv3x3'):
  """Implement torch's padding."""
  _d =  max(1, dilation)
  x = jnp.pad(x, [(0, 0), (_d, _d), (_d, _d), (0, 0)], 'constant', (0, 0))
  return nn.Conv(features, (3, 3), strides, padding='VALID', kernel_dilation=(_d, _d),
                 feature_group_count=groups, use_bias=False, name=name)(x)


class BasicBlock(nn.Module):
  features: int
  norm: Any = None
  strides: (int, int) = (1, 1)
  downsample: bool = False
  groups: int = 1
  base_width: int = 64
  dilation: int = 1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    if self.groups != 1 or self.base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if self.dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

    identity = inputs

    x = conv3x3(inputs, self.features, strides=self.strides, name='conv1')
    x = self.norm(name='bn1')(x)
    x = nn.relu(x)

    x = conv3x3(x, self.features, name='conv2')
    x = self.norm(name='bn2')(x)

    if self.downsample:
      identity = conv1x1(self.features, strides=self.strides, name='downsample_conv')(identity)
      identity = self.norm(name='downsample_bn')(identity)

    x += identity
    x = nn.relu(x)

    return x


class Bottleneck(nn.Module):
  features: int
  norm: Any
  strides: (int, int) = (1, 1)
  downsample: bool = False
  groups: int = 1
  dilation: int = 1
  base_width: int = 64
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    width = int(self.features * (self.base_width / 64.)) * self.groups
    identity = inputs

    x = conv1x1(width, name='conv1')(inputs)
    x = self.norm(name='bn1')(x)
    x = nn.relu(x)

    x = conv3x3(x, width, strides=self.strides, groups=self.groups, dilation=self.dilation, name='conv2')
    x = self.norm(name='bn2')(x)
    x = nn.relu(x)

    x = conv1x1(self.features * 4, name='conv3')(x)
    x = self.norm(name='bn3')(x)

    if self.downsample:
      identity = conv1x1(self.features * 4, strides=self.strides, name='downsample_conv')(identity)
      identity = self.norm(name='downsample_bn')(identity)

    x += identity
    x = nn.relu(x)

    return x


class Layer(nn.Module):
  block: Any
  block_size: Sequence[int]
  dilation: int
  kwargs: Dict

  @nn.compact
  def __call__(self, x):
    x = self.block(**self.kwargs, name='block1')(x)

    self.kwargs['strides'] = (1, 1)
    self.kwargs['downsample'] = False
    self.kwargs['dilation'] = self.dilation
    for i in range(1, self.block_size):
      x = self.block(**self.kwargs, name=f'block{i+1}')(x)
    return x


class Backbone(nn.Module):
  block: Any
  layers: Any
  num_classes: int = 1000
  groups: int = 1
  width_per_group: int = 64
  replace_stride_with_dilation: Any = None
  train: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    norm = functools.partial(nn.BatchNorm, use_running_average=not self.train,
                             momentum=0.9, epsilon=1e-5, dtype=self.dtype)

    if self.replace_stride_with_dilation is None:
      # each element indicates if we should replace the 2x2 stride with a dilated convolution
      self.replace_stride_with_dilation = [False, False, False]

    if len(self.replace_stride_with_dilation) != 3:
      raise ValueError("replace_stride_with_dilation should be None "
                        "or a 3-element tuple, got {}".format(self.replace_stride_with_dilation))

    x = nn.Conv(64, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
                use_bias=False, dtype=self.dtype, name='conv1')(inputs)
    x = norm(name='bn1')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=[(1,1),(1,1)])

    dilation = 1
    for i, block_size in enumerate(self.layers):
      features = 64 * 2**i
      downsample = False
      previous_dilation = dilation
      strides = (2, 2) if i > 0 else (1, 1)

      if i > 0 and self.replace_stride_with_dilation[i-1]:
        dilation *= strides[0]
        strides = (1, 1)

      block_expansion = 4 if "Bottleneck" in self.block.__name__  else 1

      if strides != (1, 1) or x.shape[-1] != features * block_expansion:
        downsample = True

      kwargs = {
        'features': features,
        'strides': strides,
        'downsample': downsample,
        'groups': self.groups,
        'dilation': previous_dilation,
        'base_width': self.width_per_group,
        'norm': norm,
        'dtype': self.dtype,
      }

      # print(f'Layer {i+1}')
      x = Layer(self.block, block_size, dilation, kwargs, name=f'layer{i+1}')(x)

    x = x.transpose((0, 3, 1, 2))
    x = jnp.mean(x, axis=(2, 3))

    return x


class ResNet(nn.Module):
  block: Any
  layers: Any
  num_classes: int = 1000
  groups: int = 1
  width_per_group: int = 64
  replace_stride_with_dilation: Any = None
  train: bool = False
  dtype: Any = jnp.float32

  def setup(self):
    self.backbone = Backbone(self.block, self.layers, self.num_classes, self.groups,
                             self.width_per_group, self.replace_stride_with_dilation,
                             self.train, self.dtype)
    self.classifier = nn.Dense(self.num_classes, dtype=self.dtype)

  def __call__(self, inputs):
    x = self.backbone(inputs)
    x = self.classifier(x)

    return x


def _get_flax_keys(keys):
  layerblock = None
  layer_idx = None
  if len(keys) == 2:  # first layer and classifier
    layer, param = keys
  elif len(keys) == 4:  # block layer
    layerblock, block_idx, layer, param = keys
  elif len(keys) == 5:  # downsample layer
    layerblock, block_idx, layer, layer_idx, param = keys

  if layer_idx == '0':
    layer = 'downsample_conv'
  if layer_idx == '1':
    layer = 'downsample_bn'

  if param == 'weight':
    param = 'scale' if 'bn' in layer else 'kernel'
  if 'running' in param:
    param = 'mean' if 'mean' in param else 'var'
  if layer == 'fc':
    layer = 'classifier'

  if layerblock:
    return ['backbone'] + [layerblock, f'block{int(block_idx)+1}', layer, param]

  if 'classifier' != layer:
    return ['backbone'] + [layer, param]

  return [layer, param]


def _resnet(rng, arch, block, layers, pretrained, **kwargs):
  resnet = functools.partial(ResNet, block=block, layers=layers, **kwargs)

  if pretrained:
    torch_params = utils.load_torch_params(model_urls[arch])
    flax_params = FrozenDict(utils.torch_to_linen(torch_params, _get_flax_keys))
  else:
    init_batch = jnp.ones((1, 224, 224, 3), jnp.float32)
    flax_params = ResNet(block=block, layers=layers, **kwargs).init(rng, init_batch)

  return resnet, flax_params


def resnet18(rng, pretrained=True, **kwargs):
  return _resnet(rng, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def resnet34(rng, pretrained=True, **kwargs):
  return _resnet(rng, 'resnet34', BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


def resnet50(rng, pretrained=True, **kwargs):
  return _resnet(rng, 'resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(rng, pretrained=True, **kwargs):
  return _resnet(rng, 'resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


def resnet152(rng, pretrained=True, **kwargs):
  return _resnet(rng, 'resnet152', Bottleneck, [3, 8, 36, 3], pretrained, **kwargs)


def resnext50_32x4d(rng, pretrained=True, **kwargs):
  kwargs['groups'] = 32
  kwargs['width_per_group'] = 4
  return _resnet(rng, 'resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnext101_32x8d(rng, pretrained=True, **kwargs):
  kwargs['groups'] = 32
  kwargs['width_per_group'] = 8
  return _resnet(rng, 'resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


def wide_resnet50_2(rng, pretrained=True, **kwargs):
  kwargs['width_per_group'] = 64 * 2
  return _resnet(rng, 'wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def wide_resnet101_2(rng, pretrained=True, **kwargs):
  kwargs['width_per_group'] = 64 * 2
  return _resnet(rng, 'wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


def wide_resnet101_2(rng, pretrained=True, **kwargs):
  kwargs['width_per_group'] = 64 * 2
  return _resnet(rng, 'wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)

