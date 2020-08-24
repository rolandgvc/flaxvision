from flax import nn
import jax.numpy as jnp
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
  def apply(self, x, num_classes=1000, train=False, dtype=jnp.float32):
    x = nn.Dense(x, 4096, dtype=dtype)
    x = nn.relu(x)
    x = nn.dropout(x, 0.5, deterministic=not train)
    x = nn.Dense(x, 4096, dtype=dtype)
    x = nn.relu(x)
    x = nn.dropout(x, 0.5, deterministic=not train)
    x = nn.Dense(x, num_classes, dtype=dtype)
    return x


class Features(nn.Module):
  def apply(self, x, cfg, batch_norm=False, train=False, dtype=jnp.float32):
    for v in cfg:
      if v == 'M':
        x = nn.max_pool(x, (2, 2), (2, 2))
      else:
        x = nn.Conv(x, v, (3, 3), padding="SAME", dtype=dtype)
        if batch_norm:
          x = nn.BatchNorm(x, use_running_average=not train, momentum=0.1, dtype=dtype)
        x = nn.relu(x)
    return x


class VGG(nn.Module):
  def apply(self, x, rng, cfg, num_classes=1000, batch_norm=False, train=False, dtype=jnp.float32):
    x = Features(x, cfg, batch_norm, train, dtype, name='features')
    x = x.transpose((0, 3, 1, 2))
    x = x.reshape((x.shape[0], -1))
    x = Classifier(x, 1000, train, name='classifier')
    return x


def torch2jax(pt_state, cfg, batch_norm=False):
  jax_params, jax_state  = {}, {}
  conv_idx = 0
  bn_idx = 1

  tensor_iter = iter(pt_state.items())

  def next_tensor():
    _, tensor = next(tensor_iter)
    return tensor.detach().numpy()

  jax_params["features"] = {}
  for layer_cfg in cfg:
    if isinstance(layer_cfg, int):
      jax_params["features"][f"Conv_{conv_idx}"] = {
          'kernel': np.transpose(next_tensor(), (2,3,1,0)),
          'bias': next_tensor(),
      }
      conv_idx += 2 if batch_norm else 1

      if batch_norm:
        jax_params["features"][f"BatchNorm_{bn_idx}"] = {
            'scale': next_tensor(),
            'bias': next_tensor(),
        }
        jax_state[f"/features/BatchNorm_{bn_idx}"] = {
            "mean": next_tensor(),
            "var": next_tensor(),
        }
        bn_idx += 2

  jax_params["classifier"] = {}
  for idx in range(3):
    jax_params["classifier"][f"Dense_{idx}"] = {
        'kernel': np.transpose(next_tensor()),
        'bias': next_tensor(),
    }
  return jax_params, jax_state


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, rng, batch_norm, pretrained, **kwargs):
  vgg = VGG.partial(rng=rng, cfg=cfgs[cfg], batch_norm=batch_norm, **kwargs)

  if pretrained:
    pt_state = utils.load_state_dict_from_url(model_urls[arch])
    params, state = torch2jax(pt_state, cfgs[cfg], batch_norm)
  else:
    with nn.stateful() as state:
      _, params = vgg.init_by_shape(rng, [(1, 224, 224, 3)])
    state = state.as_dict()

  return nn.Model(vgg, params), state


def vgg11(rng, pretrained=True, **kwargs):
  return _vgg('vgg11', 'A', rng, False, pretrained, **kwargs)


def vgg11_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg11_bn', 'A', rng, True, pretrained, **kwargs)


def vgg13(rng, pretrained=True, **kwargs):
  return _vgg('vgg13', 'B', rng, False, pretrained, **kwargs)


def vgg13_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg13_bn', 'B', rng, False, pretrained, **kwargs)


def vgg16(rng, pretrained=True, **kwargs):
  return _vgg('vgg16', 'D', rng, False, pretrained, **kwargs)


def vgg16_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg16_bn', 'D', rng, False, pretrained, **kwargs)


def vgg19(rng, pretrained=True, **kwargs):
  return _vgg('vgg19', 'E', rng, False, pretrained, **kwargs)


def vgg19_bn(rng, pretrained=True, **kwargs):
  return _vgg('vgg19_bn', 'E', rng, True, pretrained, **kwargs)