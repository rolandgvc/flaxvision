from flax import nn
import jax
import jax.numpy as jnp
from .. import utils


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


def max_pool(x, pool_size, strides, padding):
  """Temporary fix to pooling with explicit padding"""
  padding2 = [(0, 0)] + padding + [(0, 0)]
  x = jnp.pad(x, padding2, 'constant', (0,0))
  x = nn.max_pool(x, pool_size, strides)
  return x


class DenseLayer(nn.Module):
  def apply(self, x, growth_rate, bn_size, drop_rate, train=False):
    x = jnp.concatenate(x, 3)

    x = nn.BatchNorm(x, use_running_average=not train, name='norm1')
    x = nn.relu(x)
    x = nn.Conv(x, bn_size*growth_rate, (1, 1), (1, 1), padding='VALID', bias=False, name='conv1')

    x = nn.BatchNorm(x, use_running_average=not train, name='norm2')
    x = nn.relu(x)
    x = nn.Conv(x, growth_rate, (3, 3), (1, 1), padding='SAME', bias=False, name='conv2')

    if drop_rate:
      x = nn.dropout(x, rate=drop_rate, deterministic=not train)

    return x


class DenseBlock(nn.Module):
  def apply(self, x, num_layers, bn_size, growth_rate, drop_rate, train=False):
    features = [x]
    for i in range(num_layers):
      new_features = DenseLayer(features, growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate, name=f'denselayer{i+1}')
      features.append(new_features)
    return jnp.concatenate(features, 3)


class Transition(nn.Module):
  def apply(self, x, num_output_features, train=False):
    x = nn.BatchNorm(x, use_running_average=not train, name='norm')
    x = nn.relu(x)
    x = nn.Conv(x, num_output_features, (1, 1), (1, 1), padding='VALID', bias=False, name='conv')
    x = nn.avg_pool(x, (2, 2), (2, 2))
    return x


class Features(nn.Module):
  def apply(self, x, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
            bn_size=4, drop_rate=0, train=False):
    # initblock
    x = nn.Conv(x, num_init_features, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], bias=False, name='conv0')
    x = nn.BatchNorm(x, use_running_average=not train, name='norm0')
    x = nn.relu(x)
    x = max_pool(x, (3, 3), (2, 2), padding=[(1, 1), (1, 1)])

    # denseblocks
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
      x = DenseBlock(x, num_layers, bn_size, growth_rate, drop_rate, name=f'denseblock{i+1}')
      num_features += num_layers * growth_rate

      if i != len(block_config) - 1:
        num_features = num_features // 2
        x = Transition(x, num_features, name=f'transition{i+1}')

    # finalblock
    x = nn.BatchNorm(x, use_running_average=not train, name='norm5')

    x = nn.relu(x)
    x = nn.avg_pool(x, (7, 7))

    return x


class DenseNet(nn.Module):
  def apply(self, x, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, 
            bn_size=4, drop_rate=0, num_classes=1000, train=False):
    x = Features(x, growth_rate, block_config, num_init_features, bn_size, drop_rate, train, name='features')
    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_classes, name='classifier')
    return x


def _densenet(rng, arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs):
  model = DenseNet.partial(growth_rate=growth_rate, block_config=block_config, num_init_features=num_init_features, **kwargs)

  if pretrained:
    pt_params = utils.load_state_dict_from_url(model_urls[arch])
    params, state = utils.torch2jax(pt_params, get_flax_keys)
  else:
    with nn.stateful() as state:
      _, params = model.init(rng, jnp.ones((1, 224, 224, 3)))
    state = state.as_dict()

  return nn.Model(model, params), state


def get_flax_keys(keys):
  if keys[-1] == 'weight':
    is_scale = 'norm' in keys[-2] if len(keys) < 6 else 'norm' in keys[-3]
    keys[-1] = 'scale' if is_scale else 'kernel'
  if 'running' in keys[-1]:
    keys[-1] = 'mean' if 'mean' in keys[-1] else 'var'

  # if index separated from layer, concatenate
  if keys[-2] in ('1', '2'):
    keys =  keys[:3] + [keys[3]+keys[4]] + [keys[5]]

  return keys


def densenet121(rng, pretrained=False, progress=True, **kwargs):
    return _densenet(rng, 'densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)
