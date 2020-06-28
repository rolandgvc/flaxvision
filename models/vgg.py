import torch
from flax import nn
from .utils import load_state_dict_from_url
import jax
import jax.numpy as jnp
import numpy as np
np.set_printoptions(threshold=np.inf)


__all__ = ['VGG', 'vgg11']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'}


class VGG(nn.Module):

    def apply(self, x, rng, cfg, num_classes=1000, dtype=jnp.float32):

        for v in cfg:
            if v == 'M':
                x = nn.max_pool(x, (2, 2), (2, 2))
            else:
                x = nn.Conv(x, v, (3, 3), padding='SAME', dtype=dtype)
                x = nn.relu(x)

        # x = nn.avg_pool(x, (1, 1), (1, 1)) # make feature map (7,7,512)

        x = x.reshape((x.shape[0], -1))  # input shape: (batch_size, 25088)
        x = nn.Dense(x, 4096, dtype=dtype, name="Dense_0")
        x = nn.relu(x)
        x = nn.dropout(x, 0.5, rng=rng)
        x = nn.Dense(x, 4096, dtype=dtype, name="Dense_1")
        x = nn.relu(x)
        x = nn.dropout(x, 0.5, rng=rng)
        x = nn.Dense(x, num_classes, dtype=dtype, name="Dense_2")

        return x


# backbone configurations
cfgs = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}


def _vgg(arch, cfg, rng, pretrained, **kwargs):
    vgg = VGG.partial(rng=rng, cfg=cfgs[cfg], **kwargs)

    if pretrained:
        pt_state = load_state_dict_from_url(model_urls[arch])
        params = convert_from_pytorch(pt_state)
    else:
        _, params = vgg.init_by_shape(rng, [(1, 256, 256, 3)])
    # print(jax.tree_map(np.shape, params))

    return nn.Model(vgg, params)


def vgg11(rng, pretrained=False, **kwargs):
    return _vgg('vgg11', 'A', rng, pretrained, **kwargs)


def convert_from_pytorch(pt_state):
    jax_state = {}
    index_map = {'0': 0, '3': 1, '6': 2, '8': 3,
                 '11': 4, '13': 5, '16': 6, '18': 7}

    for key, tensor in pt_state.items():
        layer, index, param = key.split(".")

        if layer == 'features':
            layer = 'Conv'
        if layer == 'classifier':
            layer = 'Dense'

        jax_key = f"{layer}_{index_map[index]}"

        if param == 'weight':
            jax_state[jax_key] = {'kernel': tensor.T}
        if param == 'bias':
            jax_state[jax_key][param] = tensor

    return jax_state
