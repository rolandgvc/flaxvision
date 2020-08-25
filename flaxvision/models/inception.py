from flax import nn
import jax.numpy as jnp
import numpy as np
from .. import utils


model_urls = {
  'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


class BasicConv(nn.Module):
  def apply(self, x, out_channels, train=False, dtype=jnp.float32, **kwargs):
    if 'padding' not in kwargs:
      kwargs['padding'] = 'VALID'
    x = nn.Conv(x, out_channels, bias=False, **kwargs, name='conv', dtype=dtype)
    x = nn.BatchNorm(x, use_running_average=not train, epsilon=0.001, name='bn', dtype=dtype)
    return nn.relu(x)


class InceptionA(nn.Module):
  def apply(self, x, pool_features, conv_block):
    branch1x1 = conv_block(x, 64, kernel_size=(1, 1), name='branch1x1')

    branch5x5 = conv_block(x, 48, kernel_size=(1, 1), name='branch5x5_1')
    branch5x5 = conv_block(branch5x5, 64, kernel_size=(5, 5),
                           padding=[(2, 2), (2, 2)], name='branch5x5_2')

    branch3x3dbl = conv_block(x, 64, kernel_size=(1, 1), name='branch3x3dbl_1')
    branch3x3dbl = conv_block(branch3x3dbl, 96, kernel_size=(3, 3),
                              padding=[(1, 1), (1, 1)], name='branch3x3dbl_2')
    branch3x3dbl = conv_block(branch3x3dbl, 96, kernel_size=(3, 3),
                              padding=[(1, 1), (1, 1)], name='branch3x3dbl_3')

    branch_pool = utils.avg_pool(x, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
    branch_pool = conv_block(branch_pool, pool_features, kernel_size=(1, 1), name='branch_pool')

    outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionB(nn.Module):
  def apply(self, x, conv_block):
    branch3x3 = conv_block(x, 384, kernel_size=(3, 3), strides=(2, 2), name='branch3x3')

    branch3x3dbl = conv_block(x, 64, kernel_size=(1, 1), name='branch3x3dbl_1')
    branch3x3dbl = conv_block(branch3x3dbl, 96, kernel_size=(3, 3),
                              padding=[(1, 1), (1, 1)], name='branch3x3dbl_2')
    branch3x3dbl = conv_block(branch3x3dbl, 96, kernel_size=(3, 3),
                              strides=(2, 2), name='branch3x3dbl_3')

    branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))

    outputs = [branch3x3, branch3x3dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionC(nn.Module):
  def apply(self, x, channels_7x7, conv_block):
    branch1x1 = conv_block(x, 192, kernel_size=(1, 1), name='branch1x1')

    c7 = channels_7x7
    branch7x7 = conv_block(x, c7, kernel_size=(1, 1), name='branch7x7_1')
    branch7x7 = conv_block(branch7x7, c7, kernel_size=(1, 7),
                           padding=[(0, 0), (3, 3)], name='branch7x7_2')
    branch7x7 = conv_block(branch7x7, 192, kernel_size=(7, 1),
                           padding=[(3, 3), (0, 0)], name='branch7x7_3')

    branch7x7dbl = conv_block(x, c7, kernel_size=(1, 1), name='branch7x7dbl_1')
    branch7x7dbl = conv_block(branch7x7dbl, c7, kernel_size=(7, 1),
                              padding=[(3, 3), (0, 0)], name='branch7x7dbl_2')
    branch7x7dbl = conv_block(branch7x7dbl, c7, kernel_size=(1, 7),
                              padding=[(0, 0), (3, 3)], name='branch7x7dbl_3')
    branch7x7dbl = conv_block(branch7x7dbl, c7, kernel_size=(7, 1),
                              padding=[(3, 3), (0, 0)], name='branch7x7dbl_4')
    branch7x7dbl = conv_block(branch7x7dbl, 192, kernel_size=(1, 7),
                              padding=[(0, 0), (3, 3)], name='branch7x7dbl_5')

    branch_pool = utils.avg_pool(x, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
    branch_pool = conv_block(branch_pool, 192, kernel_size=(1, 1), name='branch_pool')

    outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionD(nn.Module):
  def apply(self, x, conv_block):
    branch3x3 = conv_block(x, 192, kernel_size=(1, 1), name='branch3x3_1')
    branch3x3 = conv_block(branch3x3, 320, kernel_size=(3, 3),
                           strides=(2, 2), name='branch3x3_2')

    branch7x7x3 = conv_block(x, 192, kernel_size=(1, 1), name='branch7x7x3_1')
    branch7x7x3 = conv_block(branch7x7x3, 192, kernel_size=(1, 7),
                             padding=[(0, 0), (3, 3)], name='branch7x7x3_2')
    branch7x7x3 = conv_block(branch7x7x3, 192, kernel_size=(7, 1),
                             padding=[(3, 3), (0, 0)], name='branch7x7x3_3')
    branch7x7x3 = conv_block(branch7x7x3, 192, kernel_size=(3, 3),
                             strides=(2, 2), name='branch7x7x3_4')

    branch_pool = nn.max_pool(x, (3, 3), strides=(2, 2))

    outputs = [branch3x3, branch7x7x3, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionE(nn.Module):
  def apply(self, x, conv_block):
    branch1x1 = conv_block(x, 320, kernel_size=(1, 1), name='branch1x1')

    branch3x3 = conv_block(x, 384, kernel_size=(1, 1), name='branch3x3_1')
    branch3x3_2a = conv_block(branch3x3, 384, kernel_size=(1, 3),
                              padding=[(0, 0), (1, 1)], name='branch3x3_2a')
    branch3x3_2b = conv_block(branch3x3, 384, kernel_size=(3, 1),
                              padding=[(1, 1), (0, 0)], name='branch3x3_2b')
    branch3x3 = jnp.concatenate([branch3x3_2a, branch3x3_2b], 3)

    branch3x3dbl = conv_block(x, 448, kernel_size=(1, 1), name='branch3x3dbl_1')
    branch3x3dbl = conv_block(branch3x3dbl, 384, kernel_size=(3, 3),
                              padding=[(1, 1), (1, 1)], name='branch3x3dbl_2')
    branch3x3dbl_3a = conv_block(branch3x3dbl, 384, kernel_size=(1, 3),
                                 padding=[(0, 0), (1, 1)], name='branch3x3dbl_3a')
    branch3x3dbl_3b = conv_block(branch3x3dbl, 384, kernel_size=(3, 1),
                                 padding=[(1, 1), (0, 0)], name='branch3x3dbl_3b')
    branch3x3dbl = jnp.concatenate([branch3x3dbl_3a, branch3x3dbl_3b], 3)

    branch_pool = utils.avg_pool(x, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)])
    branch_pool = conv_block(branch_pool, 192, kernel_size=(1, 1), name='branch_pool')

    outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
    return jnp.concatenate(outputs, 3)


class InceptionAux(nn.Module):
  def apply(self, x, num_classes, conv_block):
    x = nn.avg_pool(x, (5, 5), strides=(3, 3))
    x = conv_block(x, 128, kernel_size=(1, 1), name='conv0')
    x = conv_block(x, 768, kernel_size=(5, 5), name='conv1')

    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_classes, name='fc')

    return x


class Inception(nn.Module):
  def apply(self, x, rng, num_classes=1000, aux_logits=True, train=False,
            transform_input=True, inception_blocks=None, dtype=jnp.float32):
    conv_block = BasicConv.partial(train=train, dtype=dtype)
    inception_a = InceptionA.partial(conv_block=conv_block)
    inception_b = InceptionB.partial(conv_block=conv_block)
    inception_c = InceptionC.partial(conv_block=conv_block)
    inception_d = InceptionD.partial(conv_block=conv_block)
    inception_e = InceptionE.partial(conv_block=conv_block)
    inception_aux = InceptionAux.partial(conv_block=conv_block)

    if transform_input:
      x = np.transpose(x, (0,3,1,2))
      x_ch0 = jnp.expand_dims(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
      x_ch1 = jnp.expand_dims(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
      x_ch2 = jnp.expand_dims(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
      x = jnp.concatenate((x_ch0, x_ch1, x_ch2), 1)
      x = np.transpose(x, (0,2,3,1))

    x = conv_block(x, 32, kernel_size=(3, 3), strides=(2, 2), name='Conv2d_1a_3x3')
    x = conv_block(x, 32, kernel_size=(3, 3), name='Conv2d_2a_3x3')
    x = conv_block(x, 64, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], name='Conv2d_2b_3x3')
    x = nn.max_pool(x, (3, 3), strides=(2, 2))
    x = conv_block(x, 80, kernel_size=(1, 1), name='Conv2d_3b_1x1')
    x = conv_block(x, 192, kernel_size=(3, 3), name='Conv2d_4a_3x3')
    x = nn.max_pool(x, (3, 3), strides=(2, 2))

    x = inception_a(x, pool_features=32, name='Mixed_5b')
    x = inception_a(x, pool_features=64, name='Mixed_5c')
    x = inception_a(x, pool_features=64, name='Mixed_5d')
    x = inception_b(x, name='Mixed_6a')
    x = inception_c(x, channels_7x7=128, name='Mixed_6b')
    x = inception_c(x, channels_7x7=160, name='Mixed_6c')
    x = inception_c(x, channels_7x7=160, name='Mixed_6d')
    x = inception_c(x, channels_7x7=192, name='Mixed_6e')

    aux = inception_aux(x, num_classes, name='AuxLogits') if train and aux_logits else None

    x = inception_d(x, name='Mixed_7a')
    x = inception_e(x, name='Mixed_7b')
    x = inception_e(x, name='Mixed_7c')
    x = nn.avg_pool(x, (8, 8))
    x = nn.dropout(x, 0.5, deterministic=not train, rng=rng)

    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(x, num_classes, name='fc')

    return x, aux


def _get_flax_keys(keys):
  if keys[-1] == 'weight':
    keys[-1] = 'scale' if 'bn' in keys[-2] else 'kernel'
  if 'running' in keys[-1]:
    keys[-1] = 'mean' if 'mean' in keys[-1] else 'var'
  return keys


def inception(rng, pretrained=True, **kwargs):
  model = Inception.partial(rng=rng, **kwargs)

  if pretrained:
    torch_params = utils.load_state_dict_from_url(model_urls['inception_v3'])
    params, state = utils.torch2flax(torch_params, _get_flax_keys)
  else:
    with nn.stateful() as state:
      _, params = model.init_by_shape(rng, [(1, 299, 299, 3)])
    state = state.as_dict()

  return nn.Model(model, params), state
