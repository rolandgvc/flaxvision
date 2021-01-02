from .._utils import IntermediateLayerGetter
from .. import resnet
from .deeplabv3 import DeepLabHead, deeplabv3_keys
from .fcn import FCNHead, fcn_keys

from typing import Any, Sequence, Dict
import functools
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from flaxvision import utils
import jax

__all__ = ['fcn_resnet50', 'fcn_resnet101', 'deeplabv3_resnet50', 'deeplabv3_resnet101']


model_urls = {
    'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth',
    'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
    'deeplabv3_resnet50_coco': 'https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth',
    'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth',
}

segm_heads = { 'fcn': (FCNHead, fcn_keys), 'deeplabv3': (DeepLabHead, deeplabv3_keys) }

class SegmentationModel(nn.Module):
    backbone_fn: Any
    classifier_fn: Any

    def setup(self):
        self.backbone = self.backbone_fn()
        self.classifier = self.classifier_fn()

    def __call__(self, inputs, train: bool = False):
        input_shape = np.shape(inputs)[1:-1]
        x = self.backbone(inputs, train=train)
        x = self.classifier(x, train=train)
        x = x.transpose((0, 3, 1, 2))

        out_shape = np.shape(x)[:2] + input_shape
        x = jax.image.resize(x, shape=out_shape, method='bilinear')
        x = np.reshape(x, np.shape(x)[1:])

        return x


def _make_model(rng, head_name, backbone_name, num_classes, **kwargs):
    resnet_model, _ = resnet.__dict__[backbone_name](rng, replace_stride_with_dilation=[False, True, True])
    backbone_fn = lambda: resnet.ResNet.make_backbone(resnet_model)
    classifier_fn = lambda: segm_heads[head_name][0](num_classes)

    return SegmentationModel(backbone_fn, classifier_fn, **kwargs)


def _load_model(rng, arch_type, backbone, pretrained, num_classes, **kwargs):
    model = _make_model(rng, arch_type, backbone, num_classes, **kwargs)

    print('passed make_model')
    if pretrained:
        arch = arch_type + '_' + backbone + '_coco'
        if arch not in model_urls:
            raise NotImplementedError('pretrained {} is not supported'.format(arch))
        else:
            get_flax_keys_fn = segm_heads[arch_type][1]
            torch_params = utils.load_torch_params(model_urls[arch])
            flax_params = FrozenDict(utils.torch_to_linen(torch_params, get_flax_keys_fn))
    else:
        init_batch = jnp.ones((1, 224, 224, 3), jnp.float32)
        flax_params = model.init(rng, init_batch)

    return model, flax_params




def fcn_resnet50(rng, pretrained=True, num_classes=21, **kwargs):
    return _load_model(rng, 'fcn', 'resnet50', pretrained, num_classes, **kwargs)


def fcn_resnet101(rng, pretrained=True, num_classes=21, **kwargs):
    return _load_model(rng, 'fcn', 'resnet101', pretrained, num_classes, **kwargs)


def deeplabv3_resnet50(rng, pretrained=True, num_classes=21, **kwargs):
    return _load_model(rng, 'deeplabv3', 'resnet50', pretrained, num_classes, **kwargs)


def deeplabv3_resnet101(rng, pretrained=True, num_classes=21, **kwargs):
    return _load_model(rng, 'deeplabv3', 'resnet101', pretrained, num_classes, **kwargs)

