# FlaxVision - JAX/Flax Computer Vision Models

**Project Type**: Python ML Library  
**Version**: 0.1.0  
**Purpose**: PyTorch to JAX/Flax computer vision model ports  
**Last Updated**: 2025-07-16

## Quick Start

```bash
# Development setup
pip install -e .
pytest  # Run tests
python -m pytest -o log_cli=true --log-cli-level=INFO  # Verbose testing

# Example usage
from flaxvision import models
import jax.random as random
rng = random.PRNGKey(0)
model, params = models.resnet50(rng, pretrained=True)
```

## Project Structure

```
flaxvision/
├── __init__.py                     # Package entry point
├── models/                         # Model implementations
│   ├── __init__.py                # Model exports registry
│   ├── densenet.py                # DenseNet121/161/169/201
│   ├── inception.py               # Inception v3
│   ├── resnet.py                  # ResNet/ResNeXt/Wide ResNet variants
│   ├── vgg.py                     # VGG11/13/16/19 (±batch norm)
│   └── segmentation/              # Segmentation models
│       ├── deeplabv3.py          # DeepLabv3 with ASPP
│       ├── fcn.py                # Fully Convolutional Network
│       └── segmentation.py       # Segmentation framework
└── utils.py                       # PyTorch→Flax conversion utils

tests/
├── run_tests.sh                   # Test runner script
├── test_models.py                 # Model output equivalence tests
├── test_pretrained.py             # Pretrained model validation
└── test_training.py               # Training mode tests

examples/
└── transfer_learning.ipynb        # Transfer learning example
```

## Available Models

### Classification Models
- **VGG**: `vgg11`, `vgg13`, `vgg16`, `vgg19` (with/without batch norm)
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **ResNeXt**: `resnext50_32x4d`, `resnext101_32x8d`
- **Wide ResNet**: `wide_resnet50_2`, `wide_resnet101_2`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`
- **Inception**: `inception_v3`

### Segmentation Models
- **FCN**: `fcn_resnet50`, `fcn_resnet101`
- **DeepLabv3**: `deeplabv3_resnet50`, `deeplabv3_resnet101`

## Architecture Patterns

### Model Factory Pattern
```python
def model_name(rng, pretrained=True, **kwargs):
    model = ModelClass(**kwargs)
    if pretrained:
        torch_params = utils.load_torch_params(model_urls[arch])
        flax_params = FrozenDict(utils.torch_to_linen(torch_params, _get_flax_keys))
    else:
        flax_params = model.init(rng, jnp.ones((1, 224, 224, 3)))
    return model, flax_params
```

### Backbone Abstraction
All models implement backbone separation for transfer learning:
```python
class Model(nn.Module):
    @staticmethod
    def make_backbone(): return Backbone(...)
    def setup(): self.backbone = self.make_backbone()
    def __call__(inputs, train=False): return self.classifier(self.backbone(inputs, train))
```

### Parameter Conversion
Each model has `_get_flax_keys()` function converting PyTorch→Flax parameter names:
- Handles NCHW→NHWC tensor layout conversion
- Separates batch norm statistics from learnable parameters
- Maps PyTorch dot notation to Flax nested dictionaries

## Core Utilities

### `utils.py` - Parameter Conversion
- **`load_torch_params(url)`**: Downloads PyTorch pretrained weights
- **`torch_to_linen(torch_params, get_flax_keys)`**: Converts to Linen format
- **`torch_to_flax(torch_params, get_flax_keys)`**: Converts to original Flax format

### Key Transformations
- **Convolution weights**: `(out, in, h, w)` → `(h, w, in, out)`
- **Batch normalization**: Separates running stats from learnable params
- **Parameter structure**: Flat PyTorch dict → nested Flax dict

## Testing Infrastructure

### Test Structure
- **`test_models.py`**: Output equivalence between PyTorch/Flax (tolerance: 0.0001)
- **`test_pretrained.py`**: Pretrained model validation
- **`test_training.py`**: Training mode functionality

### Test Pattern
```python
# Load both PyTorch and Flax models
torch_model = torch_models.resnet50(pretrained=True)
flax_model, flax_params = flax_models.resnet50(rng, pretrained=True)

# Compare outputs
torch_out = torch_model(inputs)
flax_out = flax_model.apply(flax_params, inputs)
assert jnp.mean(jnp.abs(torch_out - flax_out)) < 0.0001
```

## Development Workflow

### Build System
- **Package management**: `setup.py` with setuptools
- **Dependencies**: `jax>=0.2.4`, `flax>=0.3.0`, `torch>=1.4.0`
- **Testing**: `pytest` with coverage support
- **Formatting**: `yapf` code formatter

### Model Development Pattern
1. Implement model architecture in Flax/Linen
2. Create `_get_flax_keys()` parameter mapping function
3. Add factory function with pretrained weights support
4. Export in `models/__init__.py`
5. Add tests in `test_models.py`

## Key Files

### Configuration
- **`setup.py`**: Package metadata, dependencies, build config
- **`tests/run_tests.sh`**: Test execution script

### Documentation
- **`README.md`**: Project overview and quickstart
- **`CHANGELOG.md`**: Version history
- **`CONTRIBUTING.md`**: Development guidelines
- **`examples/transfer_learning.ipynb`**: Usage example

## Transfer Learning Usage

```python
# Load pretrained model
model, params = models.resnet50(rng, pretrained=True)

# Extract backbone for transfer learning
backbone = model.make_backbone()

# Create new classifier for your task
class TransferModel(nn.Module):
    num_classes: int
    def setup(self):
        self.backbone = backbone
        self.classifier = nn.Dense(self.num_classes)
    def __call__(self, x, train=False):
        features = self.backbone(x, train)
        return self.classifier(features)
```

## Common Issues & Solutions

### Input Format
- **Expected**: NHWC format (Batch, Height, Width, Channels)
- **Inception v3**: Requires 299x299 input size (others use 224x224)

### Training Mode
- Use `train=True` parameter for proper batch norm and dropout behavior
- Separate batch statistics in `batch_stats` from learnable parameters

### Memory Management
- Explicitly delete models after testing to prevent OOM
- Use `del model` in test cleanup

## Repository Information

- **GitHub**: https://github.com/rolandgvc/flaxvision
- **License**: Apache 2.0
- **Author**: Roland Gavrilescu
- **Status**: Active development (API subject to change)