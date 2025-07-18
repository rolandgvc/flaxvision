# Flaxvision

Flaxvision is a neural network model library that ports popular computer vision models from PyTorch's torchvision to JAX & Flax. The project focuses on enabling transfer learning and high-performance inference using Google's JAX ecosystem while maintaining compatibility with PyTorch pretrained weights.

## Project Structure

```
flaxvision/
├── examples/
│   └── transfer_learning.ipynb       # Transfer learning example notebook
├── flaxvision/                       # Main package directory
│   ├── __init__.py
│   ├── models/                       # Neural network model implementations
│   │   ├── __init__.py
│   │   ├── densenet.py              # DenseNet architecture variants
│   │   ├── inception.py             # Inception v3 model
│   │   ├── resnet.py                # ResNet and ResNext variants
│   │   ├── vgg.py                   # VGG architecture variants
│   │   └── segmentation/            # Image segmentation models
│   │       ├── __init__.py
│   │       ├── deeplabv3.py         # DeepLabv3 segmentation model
│   │       ├── fcn.py               # Fully Convolutional Network
│   │       └── segmentation.py      # Segmentation model factory
│   └── utils.py                     # PyTorch to Flax parameter conversion utilities
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── run_tests.sh                 # Test runner script
│   ├── test_models.py               # Model architecture tests
│   ├── test_pretrained.py           # Pretrained model tests
│   └── test_training.py             # Training mode tests
├── setup.py                         # Package configuration
├── README.md                        # Project overview
├── CONTRIBUTING.md                  # Contribution guidelines
├── CHANGELOG.md                     # Version history
└── LICENSE                          # MIT License
```

## Build & Commands

### Development Setup
- Install in development mode: `pip install -e .`
- Install with testing dependencies: `pip install -e .[testing]`
- Standard installation: `pip install -e .[testing]`

### Testing
- Run all tests: `pytest`
- Run tests with detailed logging: `python -m pytest -o log_cli=true --log-cli-level=INFO`
- Run tests via script: `./tests/run_tests.sh`
- Run specific test file: `python -m unittest tests.test_models`

### Code Quality
- Format code: `yapf --in-place --recursive .`
- Check formatting before commits: `yapf --diff --recursive .`

### Development Environment
- Python 3.6+ required
- JAX for high-performance computing
- Flax (Linen API) for neural network modules
- PyTorch for pretrained weight loading

## Code Style

### Python Conventions
- **Indentation**: 2 spaces consistently
- **Imports**: Grouped by type (standard library, third-party, local), absolute imports preferred
- **Naming**: 
  - Classes: PascalCase (`BasicBlock`, `ResNet`, `DenseLayer`)
  - Functions: snake_case (`dilated_conv3x3`, `torch_to_flax`)
  - Variables: snake_case (`model_urls`, `num_classes`)
  - Constants: UPPER_SNAKE_CASE (`MODELS_LIST`, `RNG`)
- **Type Hints**: Extensive use of typing annotations
- **Documentation**: Minimal docstrings, inline comments for complex operations

### Import Style
```python
from typing import Any, Sequence, Dict, Optional
from functools import partial
from flax import linen as nn
from flax.core import FrozenDict
import jax.numpy as jnp
import numpy as np
from .. import utils
```

### Module Structure
- Each model follows Flax Module pattern with `@nn.compact` decorator
- Factory functions for model creation (e.g., `resnet50()`, `vgg16()`)
- Backbone + classifier separation for transfer learning
- Parameter conversion utilities for PyTorch compatibility

## Testing

### Testing Framework
- **Primary**: pytest with unittest.TestCase inheritance
- **Coverage**: pytest-cov for coverage reporting
- **Test Types**: Unit tests, model comparison tests, integration tests

### Test Categories
- **Model Output Tests**: Compare Flax vs PyTorch model outputs (tolerance: 0.0001)
- **Pretrained Model Tests**: Validate pretrained model loading and inference
- **Training Mode Tests**: Verify training mode functionality
- **Architecture Tests**: Test model construction and parameter shapes

### Tested Models
- VGG variants (11, 13, 16, 19 with/without batch normalization)
- ResNet variants (18, 34, 50, 101, 152, ResNeXt, Wide ResNet)
- Inception v3
- DenseNet variants (121, 161, 169, 201)
- Segmentation models (FCN, DeepLabv3)

## Architecture

### Core Technologies
- **JAX**: High-performance numerical computing with XLA compilation
- **Flax (Linen)**: Neural network library built on JAX
- **PyTorch**: Pretrained weight loading and reference implementations
- **NumPy**: Numerical operations and array manipulations

### Design Patterns
- **Functional Programming**: Immutable parameters, pure functions
- **Composition**: Models built from reusable components
- **Factory Pattern**: Model creation through factory functions
- **Parameter Conversion**: PyTorch to Flax weight conversion utilities

### Key Features
- Transfer learning support with pretrained weights
- Image segmentation models (DeepLabv3, FCN)
- Dilated convolutions for ResNet models
- Backbone extraction for feature extraction
- Cross-framework compatibility validation

## Security

### Best Practices
- No hardcoded secrets or API keys in codebase
- Model weights loaded from trusted sources (PyTorch Hub)
- Input validation through JAX/Flax type system
- Immutable parameter structures prevent accidental modification
- Testing ensures model behavior consistency

### Dependencies
- Use pinned minimum versions for critical dependencies
- JAX >= 0.2.4, Flax >= 0.3.0 for stability
- Regular updates following JAX/Flax compatibility guidelines

## Git Workflow

### Development Process
- **Format code** with `yapf --in-place --recursive .` before commits
- **Run tests** with `pytest` to ensure all tests pass
- **Check formatting** with `yapf --diff --recursive .`
- Create meaningful commit messages describing changes

### Branch Strategy
- Main development on `master` branch
- Feature branches for significant changes
- Pull requests require code formatting and passing tests

### Recent Development Focus
- Transfer learning example implementation
- Linen API migration completion
- Segmentation model additions
- Test coverage improvements

## Configuration

### Package Configuration
- **setup.py**: Package metadata, dependencies, and extras
- **Version**: 0.1.0 (active development, API may change)
- **Dependencies**: JAX, Flax, PyTorch, NumPy
- **Test Dependencies**: pytest, pytest-cov, jaxlib, torchvision

### Development Dependencies
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

All configurations are centralized in `setup.py` following setuptools conventions. The project follows Python packaging best practices with clear dependency management and testing requirements.