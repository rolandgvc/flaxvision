# flaxvision
The flaxvision package contains a selection of neural network models ported from [torchvision](https://github.com/pytorch/vision) to be used with [JAX](https://github.com/google/jax) & [Flax](https://github.com/google/flax).

**Note: flaxvision is currently in active development. API and functionality may change between releases.**

### Roadmap to v0.1.0
Planned features for the first release:
- [x] Update models to [linen API](https://github.com/google/flax/tree/0132b3f234a9868b47df491efde870bdc58e97a9/linen_examples)
- [x] Add support for transfer learning
- [x] Add support to ResNet for dilated convolutions
- [x] Port DeepLabv3 model for image segmentation

## Quickstart
### Transfer Learning Example
```python
from jax import random
from flaxvision import models

rng = random.PRNGKey(0)

pretrained_model = models.vgg16(rng, pretrained=True)

```
## How To Contribute
If interested in adding additional models or improving existent ones, please start by openning an Issue describing your idea.


## Acknowledgments
The initial work for flaxvision started during the Google Summer of Code program at Google AI under [Avital Oliver](https://github.com/avital)'s mentorship.
