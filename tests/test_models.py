import torch
import torchvision.models as torch_models
import flaxvision.models as flax_models
from flax import nn
import numpy as np
import jax.numpy as jnp
from jax import random

import unittest, os
import logging

RNG = random.PRNGKey(0)

MODELS_LIST = [
    'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34',
    'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
    'wide_resnet101_2', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3'
]


class TestModels(unittest.TestCase):

  def test_outputs(self):
    log = logging.getLogger(__name__)

    flax_input = jnp.ones((1, 224, 224, 3))
    torch_input = torch.ones([1, 3, 224, 224])
    flax_inception_input = jnp.ones((1, 299, 299, 3))
    torch_inception_input = torch.ones([1, 3, 299, 299])

    for key in MODELS_LIST:
      log.info(f'testing {key}')
      torch_model, (flax_model, flax_state) = self._get_model(key)
      torch_model.eval()

      if key == 'inception_v3':
        torch_out = torch_model(torch_inception_input).detach().numpy()
        with nn.stateful(nn.Collection(flax_state), mutable=False):
          flax_out = flax_model(flax_inception_input)
        self.assertLess(
            np.mean(np.abs(flax_out[0] - torch_out)), 0.0001,
            "PyTorch and Flax models' outputs don't match.")
      else:
        torch_out = torch_model(torch_input).detach().numpy()
        with nn.stateful(nn.Collection(flax_state), mutable=False):
          flax_out = flax_model(flax_input)
        self.assertLess(
            np.mean(np.abs(flax_out - torch_out)), 0.0001,
            "PyTorch and Flax models' outputs don't match.")

      del torch_model, flax_model, flax_state

  def _get_model(self, key):
    if key == 'vgg13':
      return (torch_models.vgg13(True), flax_models.vgg13(RNG))
    if key == 'vgg13_bn':
      return (torch_models.vgg13_bn(True), flax_models.vgg13_bn(RNG))
    if key == 'vgg16':
      return (torch_models.vgg16(True), flax_models.vgg16(RNG))
    if key == 'vgg16_bn':
      return (torch_models.vgg16_bn(True), flax_models.vgg16_bn(RNG))
    if key == 'vgg19':
      return (torch_models.vgg19(True), flax_models.vgg19(RNG))
    if key == 'vgg19_bn':
      return (torch_models.vgg19_bn(True), flax_models.vgg19_bn(RNG))
    if key == 'resnet18':
      return (torch_models.resnet18(True), flax_models.resnet18(RNG))
    if key == 'resnet34':
      return (torch_models.resnet34(True), flax_models.resnet34(RNG))
    if key == 'resnet50':
      return (torch_models.resnet50(True), flax_models.resnet50(RNG))
    if key == 'resnet101':
      return (torch_models.resnet101(True), flax_models.resnet101(RNG))
    if key == 'resnet152':
      return (torch_models.resnet152(True), flax_models.resnet152(RNG))
    if key == 'resnext50_32x4d':
      return (torch_models.resnext50_32x4d(True), flax_models.resnext50_32x4d(RNG))
    if key == 'resnext101_32x8d':
      return (torch_models.resnext101_32x8d(True), flax_models.resnext101_32x8d(RNG))
    if key == 'wide_resnet50_2':
      return (torch_models.wide_resnet50_2(True), flax_models.wide_resnet50_2(RNG))
    if key == 'wide_resnet101_2':
      return (torch_models.wide_resnet101_2(True), flax_models.wide_resnet101_2(RNG))
    if key == 'densenet121':
      return (torch_models.densenet121(True), flax_models.densenet121(RNG))
    if key == 'densenet161':
      return (torch_models.densenet161(True), flax_models.densenet161(RNG))
    if key == 'densenet169':
      return (torch_models.densenet169(True), flax_models.densenet169(RNG))
    if key == 'densenet201':
      return (torch_models.densenet201(True), flax_models.densenet201(RNG))
    if key == 'inception_v3':
      return (torch_models.inception_v3(True), flax_models.inception_v3(RNG))


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  unittest.main()
