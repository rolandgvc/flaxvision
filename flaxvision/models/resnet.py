from flax import nn
import jax.numpy as jnp
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


conv1x1 = nn.Conv.partial(kernel_size=(1, 1), padding='VALID', bias=False)

def conv3x3(x, features, strides=(1, 1), groups=1,  name='conv3x3'):
  """Same padding as in pytorch"""
  x = jnp.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], 'constant', (0,0))
  return nn.Conv(x, features, (3, 3), strides, padding='VALID', 
                 feature_group_count=groups, bias=False, name=name)


class BasicBlock(nn.Module):
  expansion = 1
  
  def apply(self, x, features, strides=(1, 1), downsample=False, groups=1,
            base_width=64, norm=None, train=False, dtype=jnp.float32):
    if norm is None:
        norm = nn.BatchNorm.partial(use_running_average=not train,
                                    momentum=0.9, epsilon=1e-5, dtype=dtype)
    identity = x

    out = conv3x3(x, features, strides=strides, name='conv1')
    out = norm(out, name='bn1')
    out = nn.relu(out)
    
    out = conv3x3(out, features, name='conv2')
    out = norm(out, name='bn2')
    
    if downsample:
      identity = conv1x1(identity, features, strides=strides, name='downsample_conv')
      identity = norm(identity, name='downsample_bn') 
    
    out += identity
    out = nn.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4
  
  def apply(self, x, features, strides=(1, 1), downsample=False, groups=1,
            base_width=64, norm=None, train=False, dtype=jnp.float32):
    if norm is None:
        norm = nn.BatchNorm.partial(use_running_average=not train,
                                    momentum=0.9, epsilon=1e-5, dtype=dtype)
    width = int(features * (base_width / 64.)) * groups
    identity = x

    out = conv1x1(x, width, name='conv1')
    out = norm(out, name='bn1')
    out = nn.relu(out)

    out = conv3x3(out, width, strides=strides, groups=groups, name='conv2')
    out = norm(out, name='bn2')
    out = nn.relu(out)
    
    out = conv1x1(out, features * 4, name='conv3')
    out = norm(out, name='bn3')
    
    if downsample:
      identity = conv1x1(identity, features * 4, strides=strides, name='downsample_conv')
      identity = norm(identity, name='downsample_bn')

    out += identity
    out = nn.relu(out)

    return out


class Layer(nn.Module):
  def apply(self, x, block, block_size, **kwargs):
    x = block(x, **kwargs, name='block1')

    kwargs['strides'] = (1, 1)
    kwargs['downsample'] = False
    
    for i in range(1, block_size):
      x = block(x, **kwargs, name=f'block{i+1}')
    
    return x


class ResNet(nn.Module):
  def apply(self, x, block, layers, num_classes=1000, groups=1,
            width_per_group=64, train=False, dtype=jnp.float32):
    norm = nn.BatchNorm.partial(use_running_average=not train,
                                momentum=0.9, epsilon=1e-5, dtype=dtype)

    x = nn.Conv(x, 64, (7, 7), (2, 2), padding=[(3, 3), (3, 3)],
                bias=False, dtype=dtype, name='conv1')
    x = norm(x, name='bn1')
    x = nn.relu(x)
    x = utils.max_pool(x, (3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

    for i, block_size in enumerate(layers):
      features = 64 * 2 ** i
      downsample = False
      strides = (2, 2) if i > 0 else (1, 1)
      
      if strides != (1, 1) or x.shape[-1] != features * block.expansion:
        downsample = True
      
      kwargs = {
          'features': features,
          'strides': strides,
          'downsample': downsample,
          'groups': groups,
          'base_width': width_per_group,
          'norm': norm,
          'dtype': dtype,
      }

      x = Layer(x, block, block_size, **kwargs, name=f"layer{i+1}")

    x = x.transpose((0, 3, 1, 2))
    x = jnp.mean(x, axis=(2, 3))
    x = nn.Dense(x, num_classes, dtype=dtype, name='fc')
    
    return x


def _get_flax_keys(keys):
  layerblock = layer_idx = ''
  if len(keys) == 2:   # init / dense layer
    layer, param = keys
  elif len(keys) == 4: # regular layer
    layerblock, block_idx, layer, param = keys
  elif len(keys) == 5: # downsample layer
    layerblock, block_idx, layer, layer_idx, param = keys

  if layer == 'downsample' and layer_idx == '0':
    layer = 'downsample_conv'
    layer_idx = ''
  if layer =='downsample' and layer_idx == '1':
    layer = 'downsample_bn'
    layer_idx = ''
  if param == 'weight':
    param = 'scale' if 'bn' in layer else 'kernel'
  if 'running' in param:
    param = 'mean' if 'mean' in param else 'var'

  if layerblock:
    return [layerblock, f'block_{int(block_idx)+1}', f'{layer}{layer_idx}', param]

  return [layer, param]


def _resnet(rng, arch, block, layers, pretrained, **kwargs):
  model = ResNet.partial(block=block, layers=layers, **kwargs)
  
  if pretrained:
    pt_params = load_state_dict_from_url(model_urls[arch])
    params, state = utils.torch2flax(pt_params, _get_flax_keys)
  else:
    with nn.stateful() as state:
      _, params = model.init_by_shape(rng, [(1, 224, 224, 3)])
    state = state.as_dict()

  return nn.Model(model, params), state


def resnet18(rng, pretrained=True, **kwargs):
  return _resnet(rng, 'resnet18', BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


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