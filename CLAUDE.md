# flaxvision

flaxvision is a Python package that provides neural network models ported from PyTorch's torchvision to work with JAX and Flax. It focuses on computer vision tasks with strong support for transfer learning and offers both classification and segmentation models. The package emphasizes compatibility between PyTorch and JAX ecosystems with automatic parameter conversion.

## Project Structure

```
flaxvision/
├── flaxvision/                      # Main package directory
│   ├── __init__.py                 # Package entry point (exports models, utils)
│   ├── models/                     # Neural network model implementations
│   │   ├── __init__.py            # Model exports (all public model functions)
│   │   ├── vgg.py                 # VGG architectures (VGG11-19, with/without BN)
│   │   ├── resnet.py              # ResNet family (ResNet18-152, ResNeXt, Wide ResNet)
│   │   ├── densenet.py            # DenseNet architectures (DenseNet121-201)
│   │   ├── inception.py           # Inception v3 model
│   │   └── segmentation/          # Segmentation models
│   │       ├── __init__.py        # Segmentation model exports
│   │       ├── segmentation.py    # Base segmentation utilities and model class
│   │       ├── fcn.py             # Fully Convolutional Networks (FCN)
│   │       └── deeplabv3.py       # DeepLabv3 segmentation models
│   └── utils.py                   # PyTorch-to-Flax parameter conversion utilities
├── tests/                          # Test suite
│   ├── test_models.py             # Model output comparison tests (JAX vs PyTorch)
│   ├── test_pretrained.py         # Pretrained model loading tests
│   ├── test_training.py           # Training functionality tests
│   └── run_tests.sh               # Test runner script
├── examples/                       # Usage examples
│   └── transfer_learning.ipynb    # Transfer learning with VGG16 on MNIST
├── setup.py                       # Package configuration and dependencies
├── .style.yapf                    # Code formatting configuration (yapf, 120 chars)
├── .github/workflows/ci.yml       # GitHub Actions CI/CD
├── README.md                      # Project overview and quickstart
├── CONTRIBUTING.md                # Development and contribution guidelines
├── CHANGELOG.md                   # Version history and release notes
└── LICENSE                        # Project license
```

## Build & Commands

### Installation Commands
- Install package: `pip install .`
- Install with testing dependencies: `pip install .[testing]`
- Install in development mode: `pip install -e .`

### Development Commands
- Run tests: `./tests/run_tests.sh`
- Run tests directly: `python -m pytest -o log_cli=true --log-cli-level=INFO`
- Format code: `yapf --style=.style.yapf --recursive --in-place flaxvision/`
- Check formatting: `yapf --style=.style.yapf --recursive --diff flaxvision/`

### Available Models
- **VGG**: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
- **DenseNet**: densenet121, densenet161, densenet169, densenet201
- **Inception**: inception_v3
- **Segmentation**: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101

### Model Usage Pattern
```python
from jax import random
from flaxvision import models

rng = random.PRNGKey(0)
model, params = models.vgg16(rng, pretrained=True)
```

## Code Style

- **Language**: Python 3.6+
- **Formatting**: YAPF with 120 character line limit
- **Type hints**: Extensive use of type annotations from `typing` module
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Imports**: Relative imports within package (`from .. import utils`)
- **Documentation**: Minimal but clear docstrings for public APIs
- **Error handling**: Appropriate validation and descriptive error messages

## Testing

- **Framework**: pytest with verbose logging enabled
- **Test types**: Model output comparison, pretrained loading, training functionality
- **Coverage**: Models tested against PyTorch equivalents for output consistency
- **CI/CD**: GitHub Actions running tests on Python 3.6, 3.7
- **Test command**: `./tests/run_tests.sh` or `python -m pytest -o log_cli=true --log-cli-level=INFO`
- **Test pattern**: Compare JAX/Flax model outputs with PyTorch equivalents within tolerance (0.0001)

## Architecture

- **Core Framework**: JAX for automatic differentiation and JIT compilation
- **Neural Networks**: Flax Linen API for model definitions
- **Parameter Conversion**: Sophisticated PyTorch-to-Flax parameter mapping
- **Model Structure**: Backbone + Classifier pattern for reusability
- **Transfer Learning**: `make_backbone()` static method for feature extraction
- **Immutable Parameters**: FrozenDict for parameter storage
- **Functional Programming**: Emphasizes pure functions and immutability

### Key Design Patterns

1. **Factory Pattern**: Model functions return (model, params) tuples
2. **Composition**: Models composed of reusable components (BasicBlock, Bottleneck)
3. **Builder Pattern**: Backbone + Classifier architecture
4. **Parameter Mapping**: Automatic conversion from PyTorch pretrained weights
5. **Modular Design**: Components designed for reuse across architectures

## Security

- **Dependencies**: All dependencies are well-established ML libraries
- **Parameter Loading**: Safe parameter conversion from PyTorch state dicts
- **No External Requests**: Models load pretrained weights through torch.hub
- **Input Validation**: Appropriate validation of model inputs and parameters
- **Immutable State**: Uses FrozenDict for parameter immutability

## Git Workflow

- **Testing**: ALWAYS run `./tests/run_tests.sh` before committing
- **Formatting**: Use `yapf --style=.style.yapf` for code formatting
- **CI**: GitHub Actions runs tests on push/PR to master
- **Branching**: Feature branches merged to master
- **Version**: Currently v0.1.0 with API stability warnings

## Configuration

### Dependencies
- **Core**: numpy, jax>=0.2.4, flax>=0.3.0, torch>=1.4.0
- **Testing**: jaxlib, torchvision, pytest, pytest-cov
- **Formatting**: yapf with 120 character column limit

### Model Configuration
- **Input format**: JAX arrays in NHWC format (batch, height, width, channels)
- **PyTorch compatibility**: Automatic tensor dimension reordering
- **Pretrained weights**: Loaded from PyTorch Hub and converted to Flax
- **Training mode**: Explicit `train` parameter for dropout/batch norm behavior

## Transfer Learning Support

All models support transfer learning through:
- **make_backbone()**: Static method returns feature extraction model
- **Pretrained weights**: Automatic loading from PyTorch models
- **Flexible classifiers**: Easy to replace final classification layers
- **Segmentation**: FCN and DeepLabv3 models built on ResNet backbones

## Development Notes

- **API Status**: Currently in active development (v0.1.0)
- **Migration**: Updated to Flax Linen API from legacy Flax
- **Testing**: Comprehensive comparison with PyTorch equivalents
- **Performance**: Leverages JAX's JIT compilation for performance
- **Compatibility**: Maintains compatibility with PyTorch pretrained weights