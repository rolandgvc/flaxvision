import torch
import torchvision.models as torch_models
import flaxvision.models as flax_models
from flax import nn
import numpy as np
import jax.numpy as jnp
from jax import random

import unittest


rng = random.PRNGKey(0)


MODEL_LIST = [
  (torch_models.inception_v3(True), flax_models.inception(rng)),
  (torch_models.vgg11(True), flax_models.vgg11(rng)),
  (torch_models.vgg11_bn(True), flax_models.vgg11_bn(rng)),
  (torch_models.vgg13(True), flax_models.vgg13(rng)),
  (torch_models.vgg13_bn(True), flax_models.vgg13_bn(rng)),
  (torch_models.vgg16(True), flax_models.vgg16(rng)),
  (torch_models.vgg16_bn(True), flax_models.vgg16_bn(rng)),
  (torch_models.vgg19(True), flax_models.vgg19(rng)),
  (torch_models.vgg19_bn(True), flax_models.vgg19_bn(rng)),
  (torch_models.densenet121(True), flax_models.densenet121(rng)),
  (torch_models.densenet161(True), flax_models.densenet161(rng)),
  (torch_models.densenet169(True), flax_models.densenet169(rng)),
  (torch_models.densenet201(True), flax_models.densenet201(rng)),
  (torch_models.resnet18(True), flax_models.resnet18(rng)),
  (torch_models.resnet34(True), flax_models.resnet34(rng)),
  (torch_models.resnet50(True), flax_models.resnet50(rng)),
  (torch_models.resnet101(True), flax_models.resnet101(rng)),
  (torch_models.resnet152(True), flax_models.resnet152(rng)),
  (torch_models.resnext50_32x4d(True), flax_models.resnext50_32x4d(rng)),
  (torch_models.resnext101_32x8d(True), flax_models.resnext101_32x8d(rng)),
  (torch_models.wide_resnet50_2(True), flax_models.wide_resnet50_2(rng)),
  (torch_models.wide_resnet101_2(True), flax_models.wide_resnet101_2(rng)),
]


class TestModels(unittest.TestCase):

  def test_outputs(self):
    flax_input = jnp.ones((1, 224, 224, 3))
    torch_input = torch.ones([1, 3, 224, 224])

    flax_inception_input = jnp.ones((1, 299, 299, 3))
    torch_inception_input = torch.ones([1, 3, 299, 299])

    for i, (torch_model, flax_model) in enumerate(MODEL_LIST):
      torch_model.eval()
      flax_model, flax_state = flax_model
      if i == 0:
        torch_out = torch_model(torch_inception_input).detach().numpy()
        with nn.stateful(nn.Collection(flax_state), mutable=False):
          flax_out = flax_model(flax_inception_input)
        self.assertLess(np.mean(np.abs(flax_out[0] - torch_out)), 0.0001, "PyTorch and Flax models' outputs don't match.")
      else:
        torch_out = torch_model(torch_input).detach().numpy()
        with nn.stateful(nn.Collection(flax_state), mutable=False):
          flax_out = flax_model(flax_input)
        self.assertLess(np.mean(np.abs(flax_out - torch_out)), 0.0001, "PyTorch and Flax models' outputs don't match.")


if __name__ == '__main__':
    unittest.main()
