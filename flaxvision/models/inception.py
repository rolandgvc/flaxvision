from typing import Any, Sequence, Optional, Tuple, Union
from functools import partial
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from .. import utils

model_urls = {
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class BasicConv(nn.Module):
  out_channels: int
  kernel_size: Sequence[int]
  strides: Optional[Sequence[int]] = None
  padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID'
  train: bool = False
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.out_channels, kernel_size=self.kernel_size, strides=self.strides,
                padding=self.padding, use_bias=False, name='conv', dtype=self.dtype)(x)
    x = nn.BatchNorm(use_running_average=not self.train, epsilon=0.001, name='bn', dtype=self.dtype)(x)
    return nn.relu(x)


class InceptionA(nn.Module):
  pool_features: int
  conv_block: Any

  @nn.compact
  def __call__(self, x):
    branch1x1 = self.conv_block(64, kernel_size=(1, 1), name='branch1x1')(x)

    branch5x5 = self.conv_block(48, kernel_size=(1, 1), name='branch5x5_1')(x)
    branch5x5 = self.conv_block(64, kernel_size=(5, 5), padding=[(2, 2), (2, 2)], name='branch5x5_2')(branch5x5)

    branch3x3dbl = self.conv_block(64, kernel_size=(1, 1), name='branch3x3dbl_1')(x)
    branch3x3dbl = self.conv_block(96, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name='branch3x3dbl_2')(branch3x3dbl)
    branch3x3dbl = self.conv_block(96, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name='branch3x3dbl_3')(branch3x3dbl)

    branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
    branch_pool = self.conv_block(self.pool_features, kernel_size=(1, 1), name='branch_pool')(branch_pool)

    outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionB(nn.Module):
  conv_block: Any

  @nn.compact
  def __call__(self, x):
    branch3x3 = self.conv_block(384, kernel_size=(3, 3), strides=(2, 2), name='branch3x3')(x)

    branch3x3dbl = self.conv_block(64, kernel_size=(1, 1), name='branch3x3dbl_1')(x)
    branch3x3dbl = self.conv_block(96, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name='branch3x3dbl_2')(branch3x3dbl)
    branch3x3dbl = self.conv_block(96, kernel_size=(3, 3), strides=(2, 2), name='branch3x3dbl_3')(branch3x3dbl)

    branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))

    outputs = [branch3x3, branch3x3dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionC(nn.Module):
  channels_7x7: int
  conv_block: Any

  @nn.compact
  def __call__(self, x):
    branch1x1 = self.conv_block(192, kernel_size=(1, 1), name='branch1x1')(x)

    c7 = self.channels_7x7
    branch7x7 = self.conv_block(c7, kernel_size=(1, 1), name='branch7x7_1')(x)
    branch7x7 = self.conv_block(c7, kernel_size=(1, 7), padding=[(0, 0), (3, 3)], name='branch7x7_2')(branch7x7)
    branch7x7 = self.conv_block(192, kernel_size=(7, 1), padding=[(3, 3), (0, 0)], name='branch7x7_3')(branch7x7)

    branch7x7dbl = self.conv_block(c7, kernel_size=(1, 1), name='branch7x7dbl_1')(x)
    branch7x7dbl = self.conv_block(c7, kernel_size=(7, 1), padding=[(3, 3), (0, 0)], name='branch7x7dbl_2')(branch7x7dbl)
    branch7x7dbl = self.conv_block(c7, kernel_size=(1, 7), padding=[(0, 0), (3, 3)], name='branch7x7dbl_3')(branch7x7dbl)
    branch7x7dbl = self.conv_block(c7, kernel_size=(7, 1), padding=[(3, 3), (0, 0)], name='branch7x7dbl_4')(branch7x7dbl)
    branch7x7dbl = self.conv_block(192, kernel_size=(1, 7), padding=[(0, 0), (3, 3)], name='branch7x7dbl_5')(branch7x7dbl)

    branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
    branch_pool = self.conv_block(192, kernel_size=(1, 1), name='branch_pool')(branch_pool)

    outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionD(nn.Module):
  conv_block: Any

  @nn.compact
  def __call__(self, x):
    branch3x3 = self.conv_block(192, kernel_size=(1, 1), name='branch3x3_1')(x)
    branch3x3 = self.conv_block(320, kernel_size=(3, 3), strides=(2, 2), name='branch3x3_2')(branch3x3)

    branch7x7x3 = self.conv_block(192, kernel_size=(1, 1), name='branch7x7x3_1')(x)
    branch7x7x3 = self.conv_block(192, kernel_size=(1, 7), padding=[(0, 0), (3, 3)], name='branch7x7x3_2')(branch7x7x3)
    branch7x7x3 = self.conv_block(192, kernel_size=(7, 1), padding=[(3, 3), (0, 0)], name='branch7x7x3_3')(branch7x7x3)
    branch7x7x3 = self.conv_block(192, kernel_size=(3, 3), strides=(2, 2), name='branch7x7x3_4')(branch7x7x3)

    branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))

    outputs = [branch3x3, branch7x7x3, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionE(nn.Module):
  conv_block: Any

  @nn.compact
  def __call__(self, x):
    branch1x1 = self.conv_block(320, kernel_size=(1, 1), name='branch1x1')(x)

    branch3x3 = self.conv_block(384, kernel_size=(1, 1), name='branch3x3_1')(x)
    branch3x3_2a = self.conv_block(384, kernel_size=(1, 3), padding=[(0, 0), (1, 1)], name='branch3x3_2a')(branch3x3)
    branch3x3_2b = self.conv_block(384, kernel_size=(3, 1), padding=[(1, 1), (0, 0)], name='branch3x3_2b')(branch3x3)
    branch3x3 = jnp.concatenate([branch3x3_2a, branch3x3_2b], 3)

    branch3x3dbl = self.conv_block(448, kernel_size=(1, 1), name='branch3x3dbl_1')(x)
    branch3x3dbl = self.conv_block(384, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name='branch3x3dbl_2')(branch3x3dbl)
    branch3x3dbl_3a = self.conv_block(384, kernel_size=(1, 3), padding=[(0, 0), (1, 1)], name='branch3x3dbl_3a')(branch3x3dbl)
    branch3x3dbl_3b = self.conv_block(384, kernel_size=(3, 1), padding=[(1, 1), (0, 0)], name='branch3x3dbl_3b')(branch3x3dbl)
    branch3x3dbl = jnp.concatenate([branch3x3dbl_3a, branch3x3dbl_3b], 3)

    branch_pool = nn.avg_pool(x, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
    branch_pool = self.conv_block(192, kernel_size=(1, 1), name='branch_pool')(branch_pool)

    outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionAux(nn.Module):
  num_classes: int
  conv_block: Any

  @nn.compact
  def __call__(self, x):
    x = nn.avg_pool((5, 5), strides=(3, 3))(x)
    x = self.conv_block(128, kernel_size=(1, 1), name='conv0')(x)
    x = self.conv_block(768, kernel_size=(5, 5), name='conv1')(x)

    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(self.num_classes, name='fc')(x)

    return x


class Inception(nn.Module):
  num_classes: int = 1000
  aux_logits: bool = True
  train: bool = False
  transform_input: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    conv_block = partial(BasicConv, train=self.train, dtype=self.dtype)
    inception_a = partial(InceptionA, conv_block=conv_block)
    inception_b = partial(InceptionB, conv_block=conv_block)
    inception_c = partial(InceptionC, conv_block=conv_block)
    inception_d = partial(InceptionD, conv_block=conv_block)
    inception_e = partial(InceptionE, conv_block=conv_block)
    inception_aux = partial(InceptionAux, conv_block=conv_block)

    if self.transform_input:
      x = np.transpose(x, (0, 3, 1, 2))
      x_ch0 = jnp.expand_dims(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
      x_ch1 = jnp.expand_dims(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
      x_ch2 = jnp.expand_dims(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
      x = jnp.concatenate((x_ch0, x_ch1, x_ch2), 1)
      x = np.transpose(x, (0, 2, 3, 1))

    x = conv_block(32, kernel_size=(3, 3), strides=(2, 2), name='Conv2d_1a_3x3')(x)
    x = conv_block(32, kernel_size=(3, 3), name='Conv2d_2a_3x3')(x)
    x = conv_block(64, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name='Conv2d_2b_3x3')(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2))
    x = conv_block(80, kernel_size=(1, 1), name='Conv2d_3b_1x1')(x)
    x = conv_block(192, kernel_size=(3, 3), name='Conv2d_4a_3x3')(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2))

    x = inception_a(pool_features=32, name='Mixed_5b')(x)
    x = inception_a(pool_features=64, name='Mixed_5c')(x)
    x = inception_a(pool_features=64, name='Mixed_5d')(x)
    x = inception_b(name='Mixed_6a')(x)
    x = inception_c(channels_7x7=128, name='Mixed_6b')(x)
    x = inception_c(channels_7x7=160, name='Mixed_6c')(x)
    x = inception_c(channels_7x7=160, name='Mixed_6d')(x)
    x = inception_c(channels_7x7=192, name='Mixed_6e')(x)

    aux = inception_aux(self.num_classes, name='AuxLogits')(x) \
          if self.train and self.aux_logits else None

    x = inception_d(name='Mixed_7a')(x)
    x = inception_e(name='Mixed_7b')(x)
    x = inception_e(name='Mixed_7c')(x)
    x = nn.avg_pool(x, (8, 8))
    x = nn.Dropout(0.5)(x, deterministic=not self.train)

    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(self.num_classes, name='fc')(x)

    return x, aux


def _get_flax_keys(keys):
  if keys[-1] == 'weight':
    keys[-1] = 'scale' if 'bn' in keys[-2] else 'kernel'
  if 'running' in keys[-1]:
    keys[-1] = 'mean' if 'mean' in keys[-1] else 'var'
  return keys


def inception_v3(rng, pretrained=True, **kwargs):
  inception = partial(Inception, **kwargs)

  if pretrained:
    torch_params = utils.load_torch_params(model_urls['inception_v3'])
    flax_params = FrozenDict(utils.torch_to_linen(torch_params, _get_flax_keys))
  else:
    init_batch = jnp.ones((1, 299, 299, 3), jnp.float32)
    flax_params = Inception(**kwargs).init(rng, init_batch)

  return inception, flax_params
