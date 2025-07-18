# FlaxVision Project

FlaxVision is a Python package that ports popular computer vision models from PyTorch's torchvision library to work with JAX & Flax. The project enables transfer learning and provides JAX/Flax equivalents of neural network models for computer vision tasks.

## Project Structure

```
flaxvision/
├── flaxvision/                     # Main package
│   ├── __init__.py                # Package entry point
│   ├── models/                    # Neural network model implementations
│   │   ├── __init__.py           # Model registry and factory functions
│   │   ├── resnet.py             # ResNet family (18, 34, 50, 101, 152, ResNeXt, Wide ResNet)
│   │   ├── vgg.py                # VGG family (11, 13, 16, 19 + BatchNorm variants)
│   │   ├── densenet.py           # DenseNet family (121, 161, 169, 201)
│   │   ├── inception.py          # Inception v3 model
│   │   └── segmentation/         # Image segmentation models
│   │       ├── __init__.py       
│   │       ├── segmentation.py   # Base segmentation framework
│   │       ├── fcn.py            # FCN (Fully Convolutional Network) heads
│   │       └── deeplabv3.py      # DeepLabv3 heads
│   └── utils.py                  # PyTorch-to-Flax parameter conversion utilities
├── examples/                     # Usage examples and tutorials
│   └── transfer_learning.ipynb   # Jupyter notebook showing transfer learning
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── run_tests.sh             # Test execution script
│   ├── test_models.py           # Model output validation tests
│   ├── test_pretrained.py       # Pretrained model compatibility tests
│   └── test_training.py         # Training mode functionality tests
├── .github/workflows/           # GitHub Actions CI/CD
│   └── ci.yml                   # Continuous integration configuration
├── .style.yapf                 # Code formatting configuration
├── setup.py                    # Package configuration and dependencies
├── README.md                   # Project overview and quickstart
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md               # Release history
└── LICENSE                    # Apache 2.0 license
```

## Build & Commands

- Install package: `pip install -v .`
- Install with testing dependencies: `pip install -v .[testing]`
- Run tests: `pytest` or `./tests/run_tests.sh`
- Format code: `yapf --in-place --recursive .`
- Run CI tests locally: `python -m pytest -o log_cli=true --log-cli-level=INFO`

### Development Environment

- Python: 3.6+ (CI tests on 3.6, 3.7)
- JAX backend: jaxlib (for GPU/TPU support)
- Core dependencies: JAX ≥0.2.4, Flax ≥0.3.0, PyTorch ≥1.4.0
- Development dependencies: torchvision, pytest, pytest-cov

## Code Style

- Python formatting: yapf with 120 character line limit
- Indentation: 2 spaces (unusual for Python)
- Class names: PascalCase (`ResNet`, `BasicBlock`, `SegmentationModel`)
- Function names: snake_case (`resnet50`, `load_torch_params`, `_get_flax_keys`)
- Private functions: leading underscore (`_resnet`, `_make_model`)
- Import organization: Standard imports first, then third-party, then relative imports
- Type annotations: Extensive use of typing module (`Any`, `Sequence`, `Optional`)
- Channel format: JAX uses NHWC, PyTorch uses NCHW - conversion handled automatically
- No trailing spaces, spaces around operators
- 2 blank lines between top-level definitions, 1 blank line between methods

## Testing

- Framework: pytest with unittest.TestCase base classes
- Test execution: `./tests/run_tests.sh` runs pytest with INFO logging
- Cross-framework validation: Tests compare JAX/Flax outputs against PyTorch equivalents
- Numerical precision: 0.0001 tolerance for floating-point comparisons
- Model coverage: All VGG, ResNet, DenseNet, Inception, and segmentation models
- Input handling: Standard 224x224x3 images (299x299x3 for Inception)
- Memory management: Explicit cleanup with `del` statements
- Deterministic testing: Fixed random seed `RNG = random.PRNGKey(0)`

## Architecture

- Backend: JAX for functional programming and automatic differentiation
- Frontend: Flax (Linen API) for neural network modules
- Parameter loading: PyTorch pretrained weights converted to Flax format
- Model pattern: Backbone + classifier/head architecture for transfer learning
- Parameter structure: Nested dictionaries with `params` and `batch_stats`
- Factory functions: Each model family has factory functions (e.g., `resnet50()`)
- Backbone extraction: `make_backbone()` methods for feature extraction

## Available Models

### Classification Models
- **VGG**: vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **ResNeXt**: resnext50_32x4d, resnext101_32x8d
- **Wide ResNet**: wide_resnet50_2, wide_resnet101_2
- **DenseNet**: densenet121, densenet161, densenet169, densenet201
- **Inception**: inception_v3

### Segmentation Models
- **FCN**: fcn_resnet50, fcn_resnet101
- **DeepLabv3**: deeplabv3_resnet50, deeplabv3_resnet101

## Transfer Learning

```python
from jax import random
from flaxvision import models

# Load pretrained model
rng = random.PRNGKey(0)
model, params = models.resnet50(rng, pretrained=True)

# Extract backbone for transfer learning
backbone = models.ResNet.make_backbone(model)
```

## Security

- No secrets or API keys in repository
- Parameters loaded from verified PyTorch Hub URLs
- Input validation in model constructors
- Type safety through extensive type annotations
- Regular dependency updates via CI

## Git Workflow

- Main branch: `master`
- CI runs on push/PR to master
- **ALWAYS** run tests before committing: `./tests/run_tests.sh`
- **ALWAYS** format code before committing: `yapf --in-place --recursive .`
- Follow conventional commit messages
- Use proper branch names with prefixes (e.g., `feature/`, `bugfix/`)

## Configuration

- Package configuration: `setup.py` with setuptools
- Dependencies: Core (`jax`, `flax`, `torch`) and testing (`pytest`, `torchvision`)
- Code style: `.style.yapf` with 120 character limit
- CI/CD: GitHub Actions with Python 3.6/3.7 matrix
- Model URLs: Stored in `model_urls` dictionaries within model files
- Parameter mapping: Model-specific `_get_flax_keys()` functions for PyTorch compatibility

## Development Notes

- Uses Flax Linen API (not deprecated `flax.nn`)
- Supports both `@nn.compact` and `setup()` method patterns
- Parameter conversion handles tensor transposition (PyTorch NCHW → Flax NHWC)
- Batch norm statistics handled separately from trainable parameters
- Models return (model, params) tuples from factory functions
- Segmentation models use modular head architecture
- Cross-framework numerical validation ensures mathematical correctness