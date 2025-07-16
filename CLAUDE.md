# FlaxVision Project

FlaxVision is a JAX/Flax implementation of popular computer vision models, designed to be compatible with PyTorch's torchvision. It provides pre-trained models for classification and segmentation tasks with seamless parameter conversion from PyTorch weights. The project focuses on transfer learning and model compatibility between PyTorch and JAX ecosystems.

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

### Installation
- Install package: `pip install .`
- Install with testing dependencies: `pip install .[testing]`
- Development installation: `pip install -e .`

### Testing
- Run all tests: `./tests/run_tests.sh`
- Run tests with pytest: `python -m pytest -o log_cli=true --log-cli-level=INFO`
- Run specific test file: `python -m pytest tests/test_models.py`

### Development
- Format code: `yapf --style .style.yapf --recursive --in-place flaxvision/`
- Test individual modules: `python -m unittest tests.test_models`

### CI/CD
- Automated testing on push/PR to master branch
- Python versions: 3.6, 3.7 (matrix build)
- Platform: Ubuntu latest


## Code Style

- Python package using setuptools build system
- Code formatting: yapf with 120 character line limit
- Logging: structured logging with INFO level for tests
- Memory management: explicit cleanup of large model objects
- Import conventions: standard Python import ordering
- File structure: modular organization with clear separation of concerns
- Documentation: comprehensive docstrings and inline comments

## Testing

- Testing framework: unittest with pytest execution
- Test pattern: cross-framework validation between PyTorch and Flax
- Numerical tolerance: 0.0001 for floating-point comparisons
- Model coverage: all major architectures (VGG, ResNet, Inception, DenseNet, segmentation)
- Test data: fixed random seeds for reproducibility
- Memory testing: explicit cleanup and garbage collection

## Architecture

### Core Components
- **Models**: JAX/Flax implementations of vision architectures
- **Utils**: Parameter conversion utilities (PyTorch → Flax)
- **Segmentation**: Specialized models for semantic segmentation
- **Transfer Learning**: Pre-trained weight loading and backbone extraction

### Model Design Patterns
- **Backbone-Classifier**: Two-component architecture for modularity
- **Static Factory**: `make_backbone()` methods for feature extraction
- **Modular Building Blocks**: Reusable components across architectures
- **Parameter Conversion**: Sophisticated PyTorch-to-Flax mapping

### Available Models
- **VGG**: 8 variants (VGG11/13/16/19 with/without batch normalization)
- **ResNet**: 9 variants (ResNet18/34/50/101/152, ResNeXt50/101, Wide ResNet50/101)
- **DenseNet**: 4 variants (DenseNet121/161/169/201)
- **Inception**: 1 variant (Inception v3)
- **Segmentation**: 4 variants (FCN-ResNet50/101, DeepLabv3-ResNet50/101)

### Key Features
- **JAX/Flax compatibility**: Modern JAX ecosystem integration
- **Pre-trained weights**: Automatic PyTorch weight loading
- **Transfer learning**: Easy backbone extraction for custom heads
- **Segmentation support**: FCN and DeepLabv3 implementations
- **Numerical precision**: <0.0001 error tolerance vs PyTorch

## Dependencies

### Core Requirements
- `numpy`: Array operations
- `jax>=0.2.4`: JAX framework
- `flax>=0.3.0`: Flax neural network library
- `torch>=1.4.0`: PyTorch for weight loading

### Testing Requirements
- `jaxlib`: JAX library for CPU/GPU support
- `torchvision`: PyTorch vision models for comparison
- `pytest`: Test runner
- `pytest-cov`: Coverage reporting

## Usage Patterns

### Model Instantiation
```python
import flaxvision.models as models
from jax import random

# Create model with pretrained weights
rng = random.PRNGKey(0)
model, params = models.resnet50(rng, pretrained=True)

# Use model for inference
output = model.apply(params, inputs, mutable=False)
```

### Transfer Learning
```python
# Extract backbone for custom classifier
backbone_fn = lambda: models.ResNet.make_backbone(resnet_model)
custom_model = MyModel(backbone_fn, custom_classifier)
```

### Training Mode
```python
# Enable training mode with mutable state
output, state = model.apply(params, inputs, mutable=['batch_stats'], train=True)
```

## Security

- No secrets or API keys in repository
- Automatic weight downloading from trusted PyTorch sources
- Input validation for model parameters
- Safe parameter conversion and type checking
- Memory-safe model instantiation and cleanup

## Git Workflow

- **Testing**: Always run `./tests/run_tests.sh` before committing
- **Formatting**: Use yapf for consistent code style
- **CI/CD**: Automated testing on all PRs to master
- **Branching**: Feature branches with descriptive names
- **Commits**: Clear commit messages describing changes

## Configuration

### Model Configuration
- Input shapes: 224x224x3 (most models), 299x299x3 (Inception)
- Data format: NHWC (JAX/Flax standard)
- Batch normalization: momentum=0.9, model-specific epsilon
- Dropout: deterministic parameter for training/inference control

### Parameter Management
- **Conversion pipeline**: PyTorch → Flax parameter mapping
- **State management**: params (learnable) and batch_stats (running statistics)
- **Initialization**: JAX-based random initialization for non-pretrained models

All model implementations follow consistent patterns for easy maintenance and extension.