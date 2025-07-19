# FlaxVision Project

FlaxVision is a neural network model library that ports popular computer vision models from PyTorch's torchvision to JAX & Flax. It provides pretrained models for transfer learning, feature extraction, and image segmentation tasks using JAX's functional programming approach and XLA compilation.

## Project Structure

```
flaxvision/
├── CHANGELOG.md                       # Version history and release notes
├── CONTRIBUTING.md                    # Developer contribution guidelines  
├── LICENSE                           # Project license
├── README.md                         # Main project documentation
├── setup.py                          # Python package configuration
├── examples/
│   └── transfer_learning.ipynb      # Transfer learning tutorial
├── flaxvision/                       # Main Python package
│   ├── __init__.py                  # Package initialization
│   ├── models/                      # Neural network model implementations
│   │   ├── __init__.py             # Model exports
│   │   ├── densenet.py             # DenseNet family (121/161/169/201)
│   │   ├── inception.py            # Inception v3 model
│   │   ├── resnet.py               # ResNet family (18/34/50/101/152, ResNeXt, Wide ResNet)
│   │   ├── vgg.py                  # VGG family (11/13/16/19 with/without BN)
│   │   └── segmentation/           # Segmentation models
│   │       ├── __init__.py         # Segmentation exports  
│   │       ├── deeplabv3.py        # DeepLabv3 implementation
│   │       ├── fcn.py              # Fully Convolutional Network
│   │       └── segmentation.py     # Base segmentation functionality
│   └── utils.py                     # Utility functions and PyTorch compatibility
├── tests/                            # Test suite
│   ├── __init__.py                  # Test package initialization
│   ├── run_tests.sh                 # Test execution script
│   ├── test_models.py               # Model architecture tests
│   ├── test_pretrained.py           # Pretrained model validation
│   └── test_training.py             # Training functionality tests
└── .github/
    └── workflows/
        └── ci.yml                   # GitHub Actions CI configuration
```

## Build & Commands

- Install package: `pip install -v .`
- Install with testing dependencies: `pip install -v .[testing]`
- Run all tests: `./tests/run_tests.sh` or `pytest -o log_cli=true --log-cli-level=INFO`
- Run specific test: `python -m unittest tests.test_models -v`
- Format code: `yapf --in-place --recursive .`
- Run CI locally: Install dependencies and run tests (Python 3.6+ required)

### Development Environment

- **Testing**: Uses pytest with unittest framework
- **CI/CD**: GitHub Actions (Python 3.6, 3.7 matrix)
- **Formatting**: yapf for code formatting
- **Dependencies**: JAX/Flax for models, PyTorch for compatibility

## Code Style

- Python: Snake_case for variables/functions, PascalCase for classes
- Indentation: 2 spaces (no tabs)
- Quotes: Single quotes for strings, double quotes for docstrings
- Imports: Standard library first, third-party, then relative imports with line separation
- Type hints: Extensively used throughout codebase
- Line length: Reasonable limits with multi-line for readability
- Private functions: Prefixed with underscore (_function_name)
- Constants: SCREAMING_SNAKE_CASE
- Use descriptive names: `num_classes`, `model_urls`, `block_config`

## Testing

- Framework: unittest with pytest runner
- Coverage: pytest-cov for test coverage analysis
- Test structure: Compares PyTorch and Flax model outputs for numerical equivalence
- Tolerance: 0.0001 for float comparisons
- Test data: Fixed random seeds, standard input shapes (224x224 for most, 299x299 for Inception)
- Run individual tests: `python -m unittest tests.test_<name> -v`
- CI requirements: All tests must pass before PR merge
- Logging: INFO level logging enabled in test runs

## Architecture

- Framework: JAX/Flax (Google's functional ML framework)
- Compatibility: PyTorch weight loading via utils.py
- API: Linen API (modern Flax interface)
- Models: Computer vision architectures (classification + segmentation)
- Pattern: Functional programming with immutable parameters
- Compilation: XLA compilation for performance optimization

## Security

- No secrets or API keys in repository
- Dependencies pinned with minimum versions (JAX ≥0.2.4, Flax ≥0.3.0)
- PyTorch compatibility layer uses trusted torchvision models
- Regular dependency updates recommended
- No network requests in model code (weights loaded separately)

## Git Workflow

- Main branch: `master`
- PR requirements: Tests must pass, code must be yapf formatted
- CI: Runs on Python 3.6 and 3.7 (GitHub Actions)
- Test command: `./tests/run_tests.sh`
- Format before commit: `yapf --in-place --recursive .`
- No force push to master branch

## Configuration

### Key Dependencies (setup.py):
```python
install_requires = [
    'numpy',
    'jax>=0.2.4', 
    'flax>=0.3.0',
    'torch>=1.4.0',
]

tests_require = [
    'jaxlib',
    'torchvision', 
    'pytest',
    'pytest-cov',
]
```

### Available Models:
- **ResNet**: resnet18/34/50/101/152, resnext50_32x4d/101_32x8d, wide_resnet50_2/101_2
- **VGG**: vgg11/13/16/19 (with/without batch normalization)  
- **DenseNet**: densenet121/161/169/201
- **Inception**: inception_v3
- **Segmentation**: fcn_resnet50/101, deeplabv3_resnet50/101

### Usage Patterns:
```python
# Load model with pretrained weights
from flaxvision import models
import jax.random as random

rng = random.PRNGKey(0)
model = models.resnet50(rng, pretrained=True)

# Transfer learning with backbone
backbone = models.resnet50(rng, pretrained=True, make_backbone=True)
```

All models support `make_backbone=True` for feature extraction and transfer learning tasks.