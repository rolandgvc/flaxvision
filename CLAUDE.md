# flaxvision Project

flaxvision is a Python computer vision library that ports neural network models from PyTorch's torchvision to work with JAX and Flax. It provides pretrained computer vision models optimized for the JAX ecosystem, focusing on transfer learning and research applications.

## Project Structure

```
flaxvision/
├── flaxvision/                      # Main package directory
│   ├── __init__.py                 # Package initialization, exports models and utils
│   ├── utils.py                    # PyTorch-to-Flax parameter conversion utilities
│   └── models/                     # Neural network model implementations
│       ├── __init__.py            # Model registry and exports
│       ├── vgg.py                 # VGG variants (VGG11/13/16/19 with/without BN)
│       ├── resnet.py              # ResNet family (ResNet18/34/50/101/152, ResNeXt, Wide ResNet)
│       ├── densenet.py            # DenseNet variants (DenseNet121/161/169/201)
│       ├── inception.py           # Inception v3 architecture
│       └── segmentation/          # Semantic segmentation models
│           ├── __init__.py        # Segmentation model exports
│           ├── fcn.py             # Fully Convolutional Networks
│           ├── deeplabv3.py       # DeepLabv3 architecture
│           └── segmentation.py    # Base segmentation utilities and model creation
├── tests/                          # Test suite
│   ├── test_models.py             # Model output validation tests
│   ├── test_pretrained.py         # Pretrained model loading tests
│   ├── test_training.py           # Training functionality tests
│   └── run_tests.sh               # Test execution script
├── examples/                       # Usage examples
│   └── transfer_learning.ipynb    # Transfer learning with VGG16 example
├── setup.py                       # Package configuration and dependencies
├── README.md                      # Project overview and quickstart
├── CHANGELOG.md                   # Version history
├── CONTRIBUTING.md                # Development guidelines
└── LICENSE                        # Project license
```

## Build & Commands

### Installation Commands
- Install package: `pip install .`
- Install in development mode: `pip install -e .`
- Install with test dependencies: `pip install -e .[testing]`

### Testing Commands
- Run all tests: `pytest`
- Run tests with logging: `python -m pytest -o log_cli=true --log-cli-level=INFO`
- Run tests with coverage: `pytest --cov=flaxvision`
- Use test script: `./tests/run_tests.sh`

### Code Formatting
- Format code: `yapf --in-place --recursive .`
- Check formatting: `yapf --diff --recursive .`

### Build Commands
- Build source distribution: `python setup.py sdist`
- Build wheel: `python setup.py bdist_wheel`
- Build both: `python setup.py sdist bdist_wheel`

### Development Environment
- **Python Versions**: 3.6, 3.7 (CI tested)
- **No development server**: This is a library package
- **JAX ecosystem**: Requires JAX/Flax for model execution
- **PyTorch dependency**: Required for loading pretrained weights

## Code Style

### Formatting and Structure
- **Formatter**: yapf (Yet Another Python Formatter)
- **Indentation**: 2 spaces consistently
- **Line Length**: ~100-120 characters
- **String Quotes**: Single quotes preferred
- **Import Order**: Type hints → Standard library → Third-party → Relative imports

### Naming Conventions
- **Classes**: PascalCase (`ResNet`, `VGG`, `BasicBlock`)
- **Functions**: snake_case (`resnet50()`, `vgg16()`, `make_backbone()`)
- **Private functions**: Leading underscore (`_resnet()`, `_get_flax_keys()`)
- **Constants**: UPPERCASE (`MODEL_URLS`, `MODELS_LIST`)

### Type Annotations
- Use typing module consistently: `from typing import Any, Sequence, Dict, Optional`
- Common type aliases: `ModuleDef = Any` for Flax modules
- Function signatures include parameter and return types
- JAX types: `jnp.float32`, `jnp.ndarray`

### Documentation Style
- Minimal docstrings for complex functions
- Inline comments for complex logic
- Section comments for major code blocks
- Error messages should be descriptive and actionable

## Testing

### Test Framework
- **Primary**: pytest with unittest.TestCase base classes
- **Coverage**: pytest-cov for coverage reporting
- **Logging**: Built-in logging module for test output

### Test Structure
- **Model Tests**: Compare outputs between PyTorch and Flax implementations
- **Pretrained Tests**: Validate pretrained model loading
- **Training Tests**: Test training functionality
- **Accuracy Threshold**: Mean absolute error < 0.0001 for output comparison

### Test Execution
```python
# Standard test run
pytest

# With detailed logging (CI style)
python -m pytest -o log_cli=true --log-cli-level=INFO

# Using provided script
./tests/run_tests.sh
```

### Test Data
- **Input shapes**: (1, 224, 224, 3) for most models, (1, 299, 299, 3) for Inception
- **RNG seed**: `RNG = random.PRNGKey(0)` for reproducibility
- **Model comparison**: Direct numerical comparison with PyTorch equivalents

## Architecture

### Core Technologies
- **JAX**: Google's machine learning framework for numerical computing
- **Flax**: Neural network library built on JAX (Linen API)
- **PyTorch**: Used for loading pretrained weights
- **NumPy**: Numerical operations and array handling

### Model Architecture Patterns
- **Backbone + Classifier**: Models provide `make_backbone()` method for transfer learning
- **Factory Functions**: `_resnet()`, `_vgg()`, `_densenet()` create models with shared logic
- **Parameter Conversion**: Utilities convert PyTorch weights to Flax format
- **Pretrained Loading**: Automatic download and conversion of PyTorch weights

### Key Components
1. **Model Classes**: Flax Linen modules (e.g., `ResNet`, `VGG`, `DenseNet`)
2. **Building Blocks**: Reusable components (e.g., `BasicBlock`, `Bottleneck`, `DenseLayer`)
3. **Utilities**: Parameter conversion between PyTorch and Flax formats
4. **Segmentation**: Specialized models for semantic segmentation tasks

## Available Models

### Classification Models
- **VGG**: VGG11, VGG13, VGG16, VGG19 (with/without batch normalization)
- **ResNet**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **ResNeXt**: ResNeXt50-32x4d, ResNeXt101-32x8d
- **Wide ResNet**: Wide-ResNet50-2, Wide-ResNet101-2
- **DenseNet**: DenseNet121, DenseNet161, DenseNet169, DenseNet201
- **Inception**: Inception v3

### Segmentation Models
- **FCN**: Fully Convolutional Networks with ResNet50/101 backbones
- **DeepLabv3**: DeepLabv3 with ResNet50/101 backbones

### Usage Patterns
```python
from jax import random
from flaxvision import models

rng = random.PRNGKey(0)

# Load pretrained model
model, params = models.resnet50(rng, pretrained=True)

# Create backbone for transfer learning
backbone = models.ResNet.make_backbone(
    block=models.resnet.Bottleneck,
    layers=[3, 4, 6, 3]
)

# Apply model
output = model.apply(params, input_data, mutable=False)
```

## Development Guidelines

### Parameter Conversion
- All models support loading PyTorch pretrained weights
- Parameter names are automatically converted from PyTorch to Flax format
- Weight tensors are transposed to match Flax conventions (NHWC vs NCHW)

### Adding New Models
1. Create model class inheriting from `nn.Module`
2. Implement `make_backbone()` static method for transfer learning
3. Add factory function following naming convention
4. Include parameter conversion logic
5. Add comprehensive tests comparing with PyTorch

### Error Handling
- Use descriptive error messages for validation failures
- Raise `ValueError` for invalid parameters
- Raise `NotImplementedError` for unsupported pretrained models

### Memory Management
- Models can be large; use `del` to clean up in tests
- Consider memory usage when loading multiple models
- JAX compilation can be memory-intensive

## Git Workflow

### Before Committing
1. **Format code**: `yapf --in-place --recursive .`
2. **Run tests**: `pytest` or `./tests/run_tests.sh`
3. **Check test coverage**: `pytest --cov=flaxvision`
4. **Verify model outputs**: Ensure numerical accuracy matches PyTorch

### CI/CD
- **Platform**: GitHub Actions
- **Python versions**: 3.6, 3.7
- **Test command**: `tests/run_tests.sh`
- **Triggers**: Push to master, pull requests

### Development Status
- **Version**: 0.1.0 (active development)
- **API stability**: Subject to change between releases
- **Focus**: Transfer learning, model accuracy, JAX ecosystem integration

## Configuration

### Model URLs
Pretrained weights are downloaded from PyTorch model zoo:
```python
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # ... other models
}
```

### Dependencies
- **Runtime**: numpy, jax>=0.2.4, flax>=0.3.0, torch>=1.4.0
- **Testing**: jaxlib, torchvision, pytest, pytest-cov
- **Development**: yapf for code formatting

### Environment Variables
- No specific environment variables required
- JAX configuration may be needed for GPU/TPU usage
- PyTorch models downloaded to default cache location

## Common Patterns

### Model Creation
```python
# Standard model with pretrained weights
model, params = models.resnet50(rng, pretrained=True)

# Model without pretrained weights
model, params = models.resnet50(rng, pretrained=False)

# Custom configuration
model, params = models.resnet50(rng, pretrained=True, num_classes=10)
```

### Transfer Learning
```python
# Extract backbone for feature extraction
backbone = models.ResNet.make_backbone(
    block=models.resnet.Bottleneck,
    layers=[3, 4, 6, 3]
)

# Use backbone with custom classifier
features = backbone.apply(backbone_params, input_data)
```

### Parameter Handling
```python
# Convert PyTorch parameters to Flax format
flax_params = utils.torch_to_flax(torch_params, model_keys)

# Add parameters to existing state
utils.add_to_params(params, new_params, prefix='backbone')
```

This documentation serves as the definitive guide for understanding and working with the flaxvision codebase, enabling efficient development and maintenance of JAX-based computer vision models.