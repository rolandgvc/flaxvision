import flaxvision.models as flax_models
from flax import nn
import numpy as np
import jax.numpy as jnp
from jax import random
from jax.config import config
config.enable_omnistaging()

import unittest, os
import logging

RNG = random.PRNGKey(0)

MODELS_LIST = [
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34',
    'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
    'inception_v3', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'fcn_resnet50', 'fcn_resnet101',
    'deeplabv3_resnet50', 'deeplabv3_resnet101'
]


class TestTraining(unittest.TestCase):

  def test_outputs(self):
    log = logging.getLogger(__name__)

    inputs = jnp.ones((1, 224, 224, 3))
    inception_inputs = jnp.ones((1, 299, 299, 3))

    for key in MODELS_LIST:
      log.info(f'testing inference {key}')
      model, params = self._get_model(key)
      with self.assertRaises(Exception) as context:
        try:
          if key == 'inception_v3':
            out = model.apply(params, inception_inputs, train=True, mutable=True, rngs={"params": rng, "dropout": rng})
          else:
            out = model.apply(params, inputs, train=True, mutable=True, rngs={"params": rng, "dropout": rng})
        except Exception as e:
            raise e
        self.assertTrue('Inference failed: ' in str(context.exception))

      del model, params

  def _get_model(self, key):
    if key == 'vgg11':
      return flax_models.vgg11(RNG, False)
    if key == 'vgg11_bn':
      return flax_models.vgg11_bn(RNG, False)
    if key == 'vgg13':
      return flax_models.vgg13(RNG, False)
    if key == 'vgg13_bn':
      return flax_models.vgg13_bn(RNG, False)
    if key == 'vgg16':
      return flax_models.vgg16(RNG ,False)
    if key == 'vgg16_bn':
      return flax_models.vgg16_bn(RNG, False)
    if key == 'vgg19':
      return flax_models.vgg19(RNG, False)
    if key == 'vgg19_bn':
      return flax_models.vgg19_bn(RNG, False)
    if key == 'resnet18':
      return flax_models.resnet18(RNG, False)
    if key == 'resnet34':
      return flax_models.resnet34(RNG, False)
    if key == 'resnet50':
      return flax_models.resnet50(RNG, False)
    if key == 'resnet101':
      return flax_models.resnet101(RNG, False)
    if key == 'resnet152':
      return flax_models.resnet152(RNG, False)
    if key == 'resnext50_32x4d':
      return flax_models.resnext50_32x4d(RNG, False)
    if key == 'resnext101_32x8d':
      return flax_models.resnext101_32x8d(RNG, False)
    if key == 'wide_resnet50_2':
      return flax_models.wide_resnet50_2(RNG, False)
    if key == 'wide_resnet101_2':
      return flax_models.wide_resnet101_2(RNG, False)
    if key == 'inception_v3':
      return flax_models.inception_v3(RNG, False)
    if key == 'densenet121':
      return flax_models.densenet121(RNG, False)
    if key == 'densenet161':
      return flax_models.densenet161(RNG, False)
    if key == 'densenet169':
      return flax_models.densenet169(RNG, False)
    if key == 'densenet201':
      return flax_models.densenet201(RNG, False)
    if key == 'fcn_resnet50':
      return flax_models.fcn_resnet50(RNG, False)
    if key == 'fcn_resnet101':
      return flax_models.fcn_resnet101(RNG, False)
    if key == 'deeplabv3_resnet50':
      return flax_models.deeplabv3_resnet50(RNG, False)
    if key == 'deeplabv3_resnet101':
      return flax_models.deeplabv3_resnet101(RNG, False)

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  unittest.main()
