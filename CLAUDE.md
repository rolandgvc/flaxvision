# FlaxVision - Computer Vision Models for JAX/Flax

**Project Type**: Python Library  
**Version**: 0.1.0  
**Framework**: JAX/Flax (Linen API)  
**Purpose**: PyTorch-compatible computer vision models for JAX ecosystem  
**Last Updated**: 2025-07-16

## Quick Start

```bash
# Install dependencies
pip install -e ".[testing]"

# Run tests
python -m pytest -o log_cli=true --log-cli-level=INFO

# Format code
yapf --in-place --recursive .
```

## Project Structure

```
flaxvision/
├── __init__.py                    # Main package exports
├── models/                        # Model implementations
│   ├── __init__.py               # Public API: all model functions
│   ├── resnet.py                 # ResNet family (18/34/50/101/152, ResNeXt, Wide)
│   ├── vgg.py                    # VGG models (11/13/16/19, with/without BN)
│   ├── densenet.py               # DenseNet (121/161/169/201)
│   ├── inception.py              # Inception v3 with auxiliary classifiers
│   └── segmentation/             # Semantic segmentation models
│       ├── __init__.py           # Segmentation API
│       ├── segmentation.py       # SegmentationModel base class
│       ├── fcn.py                # FCN head implementation
│       └── deeplabv3.py          # DeepLabv3 ASPP head
├── utils.py                      # PyTorch→Flax conversion utilities
tests/
├── run_tests.sh                  # Test runner script
├── test_models.py                # Model output correctness tests
├── test_pretrained.py            # Pretrained model validation
└── test_training.py              # Training mode tests (has bug)
examples/
└── transfer_learning.ipynb       # Transfer learning tutorial
```

## Architecture Overview

### Core Design Pattern

All models follow a consistent **backbone + classifier** architecture:

```python
class Model(nn.Module):
    @staticmethod
    def make_backbone(self):
        return Backbone(...)  # Feature extraction only
    
    def setup(self):
        self.backbone = Model.make_backbone(self)
        self.classifier = nn.Dense(self.num_classes)
    
    def __call__(self, inputs, train=False):
        x = self.backbone(inputs, train)
        return self.classifier(x)
```

### Available Models

#### Classification Models
- **ResNet**: `resnet18/34/50/101/152`, `resnext50_32x4d/101_32x8d`, `wide_resnet50_2/101_2`
- **VGG**: `vgg11/13/16/19` (with optional batch norm variants)
- **DenseNet**: `densenet121/161/169/201`
- **Inception**: `inception_v3` (with auxiliary classifiers)

#### Segmentation Models
- **FCN**: `fcn_resnet50/101` (simple conv head)
- **DeepLabv3**: `deeplabv3_resnet50/101` (ASPP head)

### Key Features

1. **Pretrained Weight Support**: Automatic PyTorch→Flax conversion
2. **Transfer Learning**: `make_backbone()` API for feature extraction
3. **Dilation Support**: ResNet models support dilated convolutions
4. **Unified Interface**: All models return `(model, params)` tuples

## Usage Patterns

### Basic Model Usage

```python
from jax import random
from flaxvision import models

# Load pretrained model
rng = random.PRNGKey(0)
model, params = models.resnet50(rng, pretrained=True)

# Inference
output = model.apply(params, input_batch, train=False)
```

### Transfer Learning

```python
# Extract backbone for transfer learning
backbone = models.ResNet.make_backbone(model)
backbone_params = params['params']['backbone']

# Create custom classifier
custom_model = CustomClassifier(backbone, num_classes=10)
```

### Segmentation

```python
# Load segmentation model
model, params = models.deeplabv3_resnet50(rng, pretrained=True, num_classes=21)

# Inference returns logits at input resolution
logits = model.apply(params, input_batch)
```

## Development Workflow

### Dependencies
- **Core**: `numpy`, `jax>=0.2.4`, `flax>=0.3.0`, `torch>=1.4.0`
- **Testing**: `jaxlib`, `torchvision`, `pytest`, `pytest-cov`

### Testing
```bash
# Run all tests
pytest

# Run specific test file
python -m pytest tests/test_models.py -v
```

### Code Style
```bash
# Format code before PR
yapf --in-place --recursive .
```

## Key Implementation Details

### Parameter Conversion Pipeline

1. **Download**: PyTorch pretrained weights via `torch.hub`
2. **Map Keys**: Model-specific `_get_flax_keys()` functions
3. **Convert**: `utils.torch_to_linen()` handles format conversion
4. **Structure**: Creates `{'params': {...}, 'batch_stats': {...}}` format

### Weight Transposition Rules
- **Conv layers**: `(out, in, H, W)` → `(H, W, in, out)`
- **Dense layers**: `(out, in)` → `(in, out)`
- **BatchNorm**: `weight` → `scale`, `running_mean/var` → `batch_stats`

### Segmentation Architecture
- **Backbone**: ResNet with dilated convolutions `[False, True, True]`
- **Heads**: FCN (simple) or DeepLabv3 (ASPP multi-scale)
- **Output**: Bilinear upsampling to input resolution

## Common Issues & Solutions

### Known Bugs
- `tests/test_training.py:36,38` - undefined `rng` variable (should be `RNG`)
- `examples/transfer_learning.ipynb` - syntax errors and incomplete implementation

### Performance Notes
- Input format: NHWC (JAX convention)
- Batch normalization: Proper train/inference mode handling
- Memory: Efficient parameter conversion with minimal overhead

## Contributing

1. **Issues**: Open issue before adding new models
2. **Testing**: Ensure all tests pass
3. **Formatting**: Use `yapf` code formatting
4. **PR**: Include test coverage for new features

## File Locations Reference

- Model implementations: `flaxvision/models/`
- PyTorch conversion: `flaxvision/utils.py`
- Test cases: `tests/test_*.py`
- Examples: `examples/transfer_learning.ipynb`
- Configuration: `setup.py`

## External Dependencies

- **Model weights**: PyTorch model zoo URLs
- **Conversion**: Requires PyTorch for pretrained weights
- **Testing**: Uses torchvision for reference comparisons
- **Framework**: JAX/Flax Linen API (v0.3.0+)