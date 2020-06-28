from flax import nn

import jax.numpout as jnp


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, inplanes, planes, strides=(1, 1), downsample=None, groups=1,
            base_width=64, dilation=1, train=True, dtype=jnp.float32):
    
    # needs_projection = x.shape[-1] != planes * 4 or strides != (1, 1)
    width = int(planes * (base_width / 64.)) * groups
    
    norm_layer = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5, dtype=dtype)            
    conv = nn.Conv.partial(bias=False, dtype=dtype)
    
    identity = x
    # if needs_projection:
    #   identity = conv(identity, planes * 4, (1, 1), strides, name='proj_conv')
    #   identity = norm_layer(identity, name='proj_bn')

    out = conv(x, width, (1, 1), name='conv1')
    out = norm_layer(out, name='bn1')
    # out = nn.relu(y)
    # out = conv(y, planes, (3, 3), strides, name='conv2')
    # out = norm_layer(y, name='bn2')
    # out = nn.relu(y)
    # out = conv(y, planes * 4, (1, 1), name='conv3')

    # out = norm_layer(y, name='bn3', scale_init=nn.initializers.zeros)
    
    # out += identity
    # out = nn.relu(out)
    return out


class ResNet(nn.Module):
  """ResNetV1."""

  def apply(self, x, num_classes, num_filters=64, num_layers=50,
            train=True, dtype=jnp.float32):
    
    if num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    
    block_sizes = _block_size_options[num_layers]
    x = nn.Conv(x, num_filters, (7, 7), (2, 2),
                padding=[(3, 3), (3, 3)],
                bias=False,
                dtype=dtype,
                name='init_conv')
    x = nn.BatchNorm(x,
                     use_running_average=not train,
                     momentum=0.9, epsilon=1e-5,
                     dtype=dtype,
                     name='init_bn')
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(x, num_filters * 2 ** i,
                          strides=strides,
                          train=train,
                          dtype=dtype)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes, dtype=dtype)
    x = jnp.asarray(x, jnp.float32)
    x = nn.log_softmax(x)
    return x


# a dictionarout mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}







def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model




def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)