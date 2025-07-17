# FlaxVision Project

FlaxVision is a Python library that provides computer vision models ported from PyTorch's torchvision to JAX & Flax.
The project enables transfer learning with pretrained weights and supports both image classification and segmentation tasks.

## Project Structure

```
flaxvision/
├── flaxvision/                          # Main package directory
│   ├── __init__.py                     # Package exports (models, utils)
│   ├── utils.py                        # Parameter conversion utilities for PyTorch → Flax
│   └── models/
│       ├── __init__.py                 # Model exports and factory functions
│       ├── vgg.py                      # VGG architectures (11, 13, 16, 19)
│       ├── resnet.py                   # ResNet family (18, 34, 50, 101, 152, ResNeXt, Wide ResNet)
│       ├── densenet.py                 # DenseNet architectures (121, 161, 169, 201)
│       ├── inception.py                # Inception v3 model
│       └── segmentation/
│           ├── __init__.py             # Segmentation model exports
│           ├── segmentation.py         # Base segmentation utilities and model factory
│           ├── fcn.py                  # Fully Convolutional Networks (FCN)
│           └── deeplabv3.py           # DeepLabv3 segmentation model with ASPP
├── tests/
│   ├── __init__.py                     # Test package initialization
│   ├── test_models.py                  # Model output comparison tests
│   ├── test_pretrained.py              # Pretrained model functionality tests
│   ├── test_training.py                # Training mode functionality tests
│   └── run_tests.sh                    # Test execution script
├── examples/
│   └── transfer_learning.ipynb         # Jupyter notebook with transfer learning examples
├── setup.py                            # Package configuration and dependencies
├── README.md                           # Project overview and quickstart guide
├── CONTRIBUTING.md                     # Development guidelines and contribution process
├── CHANGELOG.md                        # Version history and changes
└── LICENSE                             # Project license
```

## Build & Commands

- Install package: `pip install -v .`
- Install with test dependencies: `pip install -v .[testing]`
- Run tests: `pytest` or `./tests/run_tests.sh`
- Format code: `yapf --in-place --recursive .`
- Test with detailed logging: `python -m pytest -o log_cli=true --log-cli-level=INFO`

### Development Environment

- Python versions: 3.6, 3.7 (CI tested)
- JAX backend: Requires `jaxlib` installation
- GPU acceleration: Install appropriate JAX version for CUDA/TPU support

## Code Style

- Python formatting: `yapf` with 120 character column limit
- Indentation: 4 spaces (standard Python)
- Import organization: Standard library → Third-party → Local imports
- Naming conventions: `snake_case` for functions/variables, `PascalCase` for classes
- Type hints: Extensive use of `typing` module annotations
- Documentation: Minimal docstrings (area for improvement)
- Line length: 120 characters maximum
- Constants: `UPPER_CASE` naming convention

## Testing

- Framework: `unittest` with `pytest` runner
- Test files: `test_*.py` pattern in `/tests/` directory
- Test classes: `Test*` pattern (e.g., `TestModels`, `TestPretrained`)
- Test methods: `test_*` pattern
- Assertion pattern: `assertLess()` for numerical tolerance (threshold: 0.0001)
- Model comparison: PyTorch vs Flax output validation
- Test data: Standard 224x224x3 inputs (299x299x3 for Inception)
- Coverage: Comprehensive model testing with pretrained weights

## Architecture

- Frontend: JAX & Flax (Linen API)
- Backend: JAX numerical computing
- Parameter conversion: PyTorch → Flax utilities
- Model structure: Modular backbone + classifier pattern
- Segmentation: Specialized modules (FCN, DeepLabv3, ASPP)
- Transfer learning: Backbone extraction with `make_backbone()` methods

## Dependencies

**Core Requirements:**
- `numpy`: Array operations
- `jax>=0.2.4`: Core JAX functionality
- `flax>=0.3.0`: Neural network framework (Linen API)
- `torch>=1.4.0`: PyTorch for pretrained weight loading

**Test Requirements:**
- `jaxlib`: JAX backend
- `torchvision`: Reference implementations
- `pytest`: Testing framework
- `pytest-cov`: Coverage testing

## Model Usage Patterns

### Basic Model Creation
```python
from jax import random
from flaxvision import models

rng = random.PRNGKey(0)
model = models.resnet50(rng, pretrained=True)
```

### Transfer Learning
```python
# Extract backbone for custom classifier
backbone = models.resnet50.make_backbone(features=2048)
# Use backbone with custom head
```

### Segmentation Models
```python
# FCN with ResNet50 backbone
fcn_model = models.fcn_resnet50(rng, pretrained=True)

# DeepLabv3 with ResNet50 backbone
deeplabv3_model = models.deeplabv3_resnet50(rng, pretrained=True)
```

## Security

- No sensitive data handling in core library
- Parameter loading from trusted PyTorch sources
- Standard Python security practices
- No network requests or external API calls
- All model weights loaded from local files or standard repositories

## Git Workflow

- **ALWAYS** run `yapf --in-place --recursive .` before committing
- **ALWAYS** run `pytest` to ensure tests pass
- Follow standard GitHub workflow for pull requests
- CI automatically tests Python 3.6 and 3.7 compatibility
- Format code with yapf before creating PRs

## Configuration

- Model configuration: Defined in individual model files
- No external configuration files
- All settings hardcoded in source for simplicity
- Pretrained weights URLs defined in model modules
- JAX/Flax settings managed through framework APIs

## Common Development Tasks

- **Adding new models**: Follow existing patterns in `models/` directory
- **Parameter conversion**: Use utilities in `utils.py`
- **Testing new models**: Add tests to appropriate test files
- **Documentation**: Update README and docstrings
- **Performance optimization**: Leverage JAX JIT compilation
- **GPU support**: Ensure JAX device placement compatibility