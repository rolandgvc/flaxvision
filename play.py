import torch
import torchvision

import jax
from jax import random
import jax.numpy as jnp
from flax import nn
import numpy as np
import models

from pprint import pprint

# INPUTS
fx_input = jnp.ones((1, 224, 224, 3))
pt_input = torch.ones([1, 3, 224, 224])

pt_model = torchvision.models.vgg11_bn(True, True)
pt_model.eval()
pt_res = pt_model(pt_input).detach().numpy()

rng = random.PRNGKey(0)
fx_model, state = models.vgg11_bn(rng, pretrained=True)

with nn.stateful(nn.Collection(state), mutable=False):
  fx_res = fx_model(fx_input)

pprint(jax.tree_map(np.shape, fx_model.params))
pprint(jax.tree_map(np.shape, state))

# test output accuracy
print('Output error', np.mean(np.abs(fx_res - pt_res)))
