# Flaxvision Project

Flaxvision is a computer vision library that ports popular neural network models from PyTorch's torchvision to JAX & Flax. The project provides JAX/Flax implementations of well-known computer vision models with compatibility for pre-trained weights from PyTorch's model zoo.

## Project Structure

```
flaxvision/
├── flaxvision/                      # Main package directory
│   ├── __init__.py                 # Package initialization
│   ├── models/                     # Neural network model implementations
│   │   ├── __init__.py            # Models package initialization
│   │   ├── vgg.py                 # VGG architectures (VGG11, VGG13, VGG16, VGG19)
│   │   ├── resnet.py              # ResNet family (ResNet18-152, ResNeXt, Wide ResNet)
│   │   ├── densenet.py            # DenseNet architectures (DenseNet121, 161, 169, 201)
│   │   ├── inception.py           # Inception v3 model
│   │   └── segmentation/          # Semantic segmentation models
│   │       ├── __init__.py        # Segmentation package initialization
│   │       ├── fcn.py             # Fully Convolutional Networks
│   │       ├── deeplabv3.py       # DeepLabv3 segmentation model
│   │       └── segmentation.py    # Segmentation utilities and model factory
│   └── utils.py                   # Utility functions for PyTorch↔Flax conversion
├── tests/                          # Test suite
│   ├── __init__.py                # Test package initialization
│   ├── test_models.py             # Model output validation tests
│   ├── test_pretrained.py         # Pre-trained model loading tests
│   ├── test_training.py           # Training mode validation tests
│   └── run_tests.sh               # Test execution script
├── examples/                       # Usage examples
│   └── transfer_learning.ipynb    # Transfer learning tutorial
├── setup.py                       # Package configuration and dependencies
├── README.md                      # Project overview and quickstart
├── CONTRIBUTING.md                # Contribution guidelines
├── CHANGELOG.md                   # Version history
└── LICENSE                        # MIT License
```

## Build & Commands

- Install from source: `pip install -v .`
- Install with testing dependencies: `pip install -v .[testing]`
- Run all tests: `pytest` or `bash tests/run_tests.sh`
- Run tests with verbose logging: `python -m pytest -o log_cli=true --log-cli-level=INFO`
- Format code before PR: `yapf --in-place --recursive .`

### Development Environment

- **Python versions**: 3.6, 3.7 (officially supported)
- **JAX/Flax ecosystem**: Modern JAX arrays and Flax Linen API
- **Testing**: pytest with comprehensive model validation
- **CI/CD**: GitHub Actions for automated testing

## Code Style

- **Indentation**: 4 spaces for Python code
- **Line length**: Generally under 120 characters
- **Imports**: Grouped (stdlib, third-party, local) and alphabetically sorted
- **Naming conventions**: 
  - Classes: `PascalCase` (e.g., `BasicBlock`, `ResNet`)
  - Functions: `snake_case` (e.g., `dilated_conv3x3`, `_resnet`)
  - Variables: `snake_case` (e.g., `num_classes`, `input_shape`)
  - Constants: `UPPER_CASE` or `snake_case` for dictionaries
- **Type hints**: Extensive use of typing module (`Any`, `Sequence`, `Dict`, `Optional`, `Tuple`)
- **Code formatting**: Use `yapf --in-place --recursive .` before submitting PRs
- **Documentation**: Minimal docstrings; prefer clear, descriptive code
- **JAX/Flax patterns**: Use `@nn.compact` decorator, `setup()` + `__call__()` pattern, and `jnp` over `np`

## Testing

- **Framework**: pytest with unittest base classes
- **Test structure**: Three main test suites:
  - `test_models.py`: Model output validation against PyTorch equivalents
  - `test_pretrained.py`: Pre-trained model loading and validation
  - `test_training.py`: Training mode functionality
- **Model validation**: Cross-framework comparison with < 0.0001 tolerance
- **Coverage**: 21 different model variants tested
- **Execution**: `pytest` or `bash tests/run_tests.sh`
- **CI integration**: Automated testing on Python 3.6, 3.7 with GitHub Actions

## Architecture

- **Core framework**: JAX for numerical computing, Flax Linen for neural networks
- **Model types**: 
  - **Classification**: ResNet, VGG, DenseNet, Inception architectures
  - **Segmentation**: FCN, DeepLabv3 models with backbone support
- **Transfer learning**: All models support `make_backbone()` method for feature extraction
- **Pre-trained weights**: Automatic download and conversion from PyTorch model zoo
- **Parameter conversion**: Utilities for PyTorch ↔ Flax parameter translation

## Security

- **Dependency management**: Pinned minimum versions for core dependencies
- **No secrets**: No API keys or sensitive data in repository
- **Safe imports**: Only imports from trusted sources (Google JAX/Flax, PyTorch)
- **Model weights**: Downloaded from official PyTorch model zoo URLs
- **Code review**: All contributions require testing and formatting validation

## Git Workflow

- **Branch protection**: Master branch requires PR approval
- **Pre-commit requirements**: 
  - All tests must pass: `pytest`
  - Code must be formatted: `yapf --in-place --recursive .`
- **Testing**: Comprehensive test suite with cross-framework validation
- **CI/CD**: GitHub Actions on push and PR to master
- **Contribution process**: Open issue first, then create PR with tests

## Configuration

### Dependencies (from setup.py)

**Core dependencies:**
- `numpy`: Numerical computing foundation
- `jax>=0.2.4`: Google's machine learning framework
- `flax>=0.3.0`: Neural network library (Linen API)
- `torch>=1.4.0`: PyTorch for pre-trained weight loading

**Testing dependencies:**
- `jaxlib`: JAX XLA compilation library
- `torchvision`: PyTorch computer vision models for comparison
- `pytest`: Testing framework
- `pytest-cov`: Coverage reporting

### Available Models

**Classification Models:**
- VGG: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`
- ResNet: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- ResNeXt: `resnext50_32x4d`, `resnext101_32x8d`
- Wide ResNet: `wide_resnet50_2`, `wide_resnet101_2`
- DenseNet: `densenet121`, `densenet161`, `densenet169`, `densenet201`
- Inception: `inception_v3`

**Segmentation Models:**
- FCN: `fcn_resnet50`, `fcn_resnet101`
- DeepLabv3: `deeplabv3_resnet50`, `deeplabv3_resnet101`

### Usage Pattern

```python
from jax import random
from flaxvision import models

# Initialize random key
rng = random.PRNGKey(0)

# Load pre-trained model
model = models.resnet50(rng, pretrained=True)

# Extract backbone for transfer learning
backbone = models.resnet50.make_backbone(rng, pretrained=True)
```

All models follow consistent API patterns with support for pre-trained weights, transfer learning, and custom configurations.