# flaxvision Project

flaxvision is a Python library that ports popular computer vision models from PyTorch's torchvision to work with JAX and Flax frameworks. The project provides JAX/Flax implementations of well-known deep learning models, enabling users to leverage these models in the JAX ecosystem with pretrained weights support.

## Project Structure

```
flaxvision/
├── flaxvision/                      # Main package directory
│   ├── __init__.py                 # Package initialization (exports models, utils)
│   ├── models/                     # Neural network model implementations
│   │   ├── __init__.py            # Exports all model factory functions
│   │   ├── vgg.py                 # VGG model family (VGG11, VGG13, VGG16, VGG19)
│   │   ├── resnet.py              # ResNet family (ResNet18-152, ResNeXt, Wide ResNet)
│   │   ├── densenet.py            # DenseNet family (DenseNet121, 161, 169, 201)
│   │   ├── inception.py           # Inception v3 model implementation
│   │   └── segmentation/          # Segmentation models subdirectory
│   │       ├── __init__.py        # Segmentation models exports
│   │       ├── segmentation.py    # Base segmentation functionality
│   │       ├── fcn.py             # FCN (Fully Convolutional Network) models
│   │       └── deeplabv3.py       # DeepLabv3 segmentation models
│   └── utils.py                    # Parameter conversion utilities
├── tests/                          # Test suite
│   ├── __init__.py                # Test package initialization
│   ├── test_models.py             # Model output validation tests
│   ├── test_pretrained.py         # Pretrained model loading tests
│   ├── test_training.py           # Training mode functionality tests
│   └── run_tests.sh               # Test execution script
├── examples/                       # Usage examples
│   └── transfer_learning.ipynb    # Transfer learning demonstration
├── setup.py                       # Package installation configuration
├── README.md                      # Project documentation
├── CONTRIBUTING.md                # Development guidelines
├── CHANGELOG.md                   # Version history
└── LICENSE                        # Apache 2.0 license
```

## Build & Commands

- Run tests: `pytest` or `bash tests/run_tests.sh`
- Run with coverage: `pytest --cov=flaxvision`
- Format code: `yapf --in-place --recursive .`
- Install package: `pip install -e .`
- Install with test dependencies: `pip install -e ".[testing]"`

### Development Environment

- Python 3.x environment
- JAX/Flax for model implementation
- PyTorch for pretrained weights and comparison testing
- Jupyter notebook for examples

## Code Style

- Python code formatted with **yapf** (column limit: 120 characters)
- **snake_case** for variables and functions: `model_urls`, `dilated_conv3x3`
- **PascalCase** for classes: `BasicBlock`, `Bottleneck`, `ResNet`
- **UPPER_CASE** for constants: `MODELS_LIST`, `RNG`
- Private functions prefixed with underscore: `_resnet()`, `_get_model()`
- Extensive use of type hints with `typing` module
- Consistent import organization:
  1. Standard library imports
  2. Third-party imports (flax, jax, numpy, torch)
  3. Local imports (relative imports using `..`)
- Import aliases: `jnp` for `jax.numpy`, `nn` for `flax.linen`
- Minimal but effective inline comments
- 2-space indentation consistently

## Testing

- Uses `unittest` framework for test structure
- Test classes: `TestModels`, `TestPretrained`, `TestTraining`
- Comparison testing between PyTorch and Flax implementations
- Numerical tolerance assertions (0.0001) for output validation
- Shared `MODELS_LIST` constant across test files
- Consistent use of `RNG = random.PRNGKey(0)` for reproducibility
- Test both pretrained and randomly initialized models
- Validate inference and training modes

## Architecture

- **Framework**: JAX/Flax (Linen API) for model implementation
- **Compatibility**: PyTorch for pretrained weights and comparison
- **Dependencies**: 
  - `numpy` (numerical computing)
  - `jax>=0.2.4` (core framework)
  - `flax>=0.3.0` (neural network library)
  - `torch>=1.4.0` (pretrained weights)
- **Design patterns**:
  - Functional programming with JAX
  - Composition over inheritance
  - Factory functions for model creation
  - Backbone/classifier separation for transfer learning

## Model Implementations

### Classification Models
- **VGG**: VGG11, VGG13, VGG16, VGG19 (with optional batch normalization)
- **ResNet**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **ResNeXt**: ResNeXt50_32x4d, ResNeXt101_32x8d  
- **Wide ResNet**: Wide_ResNet50_2, Wide_ResNet101_2
- **DenseNet**: DenseNet121, DenseNet161, DenseNet169, DenseNet201
- **Inception**: Inception_v3

### Segmentation Models
- **FCN**: FCN_ResNet50, FCN_ResNet101
- **DeepLabv3**: DeepLabv3_ResNet50, DeepLabv3_ResNet101

### Transfer Learning Support
- All models include `make_backbone()` static method for feature extraction
- Supports both pretrained and randomly initialized models
- Parameter conversion utilities for PyTorch → JAX/Flax weights

## Security

- Use appropriate parameter validation in model factory functions
- Never commit API keys or sensitive data to repository
- Validate model inputs appropriately
- Use secure parameter loading from external sources
- Follow principle of least privilege for model access

## Git Workflow

- Format code with `yapf --in-place --recursive .` before committing
- Run tests with `pytest` to verify functionality
- Test both pretrained and non-pretrained models
- Ensure parameter conversion works correctly
- Follow conventional commit message format

## Configuration

### Model Factory Functions
All models follow consistent factory function patterns:
```python
def model_name(rng, pretrained=False, **kwargs):
    # Returns (model, params) tuple
```

### Parameter Conversion
- `torch_to_flax()` - Convert PyTorch parameters to Flax format
- `torch_to_linen()` - Convert PyTorch parameters to Linen format
- `load_torch_params()` - Download and convert pretrained weights

### Testing Configuration
- Test models in both inference and training modes
- Use consistent numerical tolerance (0.0001) for comparisons
- Test all available pretrained models
- Validate parameter conversion accuracy