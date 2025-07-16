# flaxvision - JAX/Flax Computer Vision Models

**Project Type:** Neural Network Model Library  
**Version:** 0.1.0  
**Framework:** JAX/Flax (ported from PyTorch/torchvision)  
**Last Updated:** 2025-07-16  

## Quick Start

```bash
# Install
pip install -v .
pip install -v .[testing]  # for development

# Run tests
bash tests/run_tests.sh

# Basic usage
from jax import random
from flaxvision import models

rng = random.PRNGKey(0)
model = models.vgg16(rng, pretrained=True)
```

## Project Structure

```
flaxvision/
├── __init__.py                  # Package initialization
├── models/                      # Neural network models
│   ├── __init__.py             # Model exports
│   ├── densenet.py             # DenseNet: 121, 161, 169, 201
│   ├── inception.py            # Inception v3
│   ├── resnet.py               # ResNet: 18, 34, 50, 101, 152 + ResNeXt + Wide
│   ├── vgg.py                  # VGG: 11, 13, 16, 19 (+ BN variants)
│   └── segmentation/           # Segmentation models
│       ├── __init__.py         # Segmentation exports
│       ├── deeplabv3.py        # DeepLabv3 implementation
│       ├── fcn.py              # FCN (Fully Convolutional Network)
│       └── segmentation.py     # fcn_resnet50/101, deeplabv3_resnet50/101
└── utils.py                    # PyTorch → Flax parameter conversion

tests/
├── __init__.py                 # Test package
├── run_tests.sh                # Test runner script
├── test_models.py              # Model inference validation
├── test_pretrained.py          # Pretrained model testing
└── test_training.py            # Training mode testing

examples/
└── transfer_learning.ipynb     # Transfer learning tutorial

setup.py                        # Package configuration
README.md                       # Project documentation
```

## Available Models

### Classification Models
- **VGG**: `vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`
- **ResNet**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- **ResNeXt**: `resnext50_32x4d`, `resnext101_32x8d`
- **Wide ResNet**: `wide_resnet50_2`, `wide_resnet101_2`
- **DenseNet**: `densenet121`, `densenet161`, `densenet169`, `densenet201`
- **Inception**: `inception_v3`

### Segmentation Models
- **FCN**: `fcn_resnet50`, `fcn_resnet101`
- **DeepLabv3**: `deeplabv3_resnet50`, `deeplabv3_resnet101`

## Architecture Patterns

### Model Structure
All models follow consistent patterns:
```python
class Model(nn.Module):
    @staticmethod
    def make_backbone(self):        # Feature extractor for transfer learning
        return Backbone(...)
    
    def setup(self):
        self.backbone = Model.make_backbone(self)
        self.classifier = nn.Dense(...)
    
    def __call__(self, inputs, train=False):
        x = self.backbone(inputs, train)
        return self.classifier(x.reshape((x.shape[0], -1)))
```

### Key Components
- **Backbone**: Feature extraction layers (convolutional)
- **Classifier**: Final classification layers (fully connected)
- **Building Blocks**: BasicBlock, Bottleneck, DenseLayer, Inception modules
- **Segmentation Heads**: FCNHead, DeepLabHead with ASPP

### Data Format Convention
- **Input**: NHWC (batch, height, width, channels)
- **Processing**: Maintains NHWC throughout
- **Pooling**: Transpose to NCHW only for global pooling

## Transfer Learning

### Extract Backbone for Custom Tasks
```python
# Get feature extractor
backbone_fn = lambda: models.VGG.make_backbone(vgg_model)

# Custom model
class CustomModel(nn.Module):
    def setup(self):
        self.backbone = backbone_fn()
        self.classifier = CustomClassifier()
```

### Freeze Backbone During Training
```python
# Frozen backbone
features = backbone(inputs, train=False)
# Trainable classifier
output = classifier(features, train=True)
```

## Parameter Conversion

### PyTorch → Flax Pipeline
1. Download PyTorch weights from official URLs
2. Convert using `utils.torch_to_linen()` function
3. Map parameter names via `_get_flax_keys()`
4. Transpose conv weights: `(out, in, h, w) → (h, w, in, out)`
5. Handle BatchNorm: `weight → scale`, `running_mean → mean`

### Key Functions (`utils.py`)
- `load_torch_params(url)`: Download PyTorch state dict
- `torch_to_linen(torch_params, get_flax_keys)`: Convert to Flax format
- `torch_to_flax()`: Legacy conversion function

## Testing

### Framework
- **Primary**: `unittest` with `pytest` runner
- **Command**: `bash tests/run_tests.sh`
- **Validation**: Cross-framework comparison with PyTorch

### Test Types
1. **Model Inference** (`test_models.py`): Output validation vs PyTorch
2. **Pretrained Loading** (`test_pretrained.py`): Parameter loading tests
3. **Training Mode** (`test_training.py`): Dropout/BatchNorm behavior

### Validation Pattern
```python
# Numerical tolerance: < 0.0001 difference
assert np.mean(np.abs(flax_out - torch_out)) < 0.0001
```

## Dependencies

### Core Runtime
- `jax >= 0.2.4` - JAX framework
- `flax >= 0.3.0` - Neural network library
- `numpy` - Numerical computing
- `torch >= 1.4.0` - PyTorch (for parameter conversion)

### Testing
- `jaxlib` - JAX linear algebra library
- `torchvision` - PyTorch vision library
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting

## Development

### Installation
```bash
pip install -v .              # Basic install
pip install -v .[testing]     # With testing dependencies
```

### Running Tests
```bash
bash tests/run_tests.sh       # All tests
python -m pytest             # pytest directly
python -m pytest tests/test_models.py  # Specific test
```

### Model Input Sizes
- **Standard Models**: 224×224×3
- **Inception v3**: 299×299×3
- **Segmentation**: Variable (auto-resize output)

## Key Files to Modify

### Adding New Models
1. **Model Implementation**: `flaxvision/models/new_model.py`
2. **Parameter Conversion**: Add `_get_flax_keys()` function
3. **Package Export**: Update `flaxvision/models/__init__.py`
4. **Testing**: Add to test files' `MODELS_LIST`

### Parameter Loading
- **URLs**: Add to `model_urls` dictionary
- **Key Mapping**: Implement `_get_flax_keys()` function
- **Conversion**: Use `torch_to_linen()` in model factory

### Testing New Models
- **Inference**: Add to `test_models.py`
- **Pretrained**: Add to `test_pretrained.py`
- **Training**: Add to `test_training.py`

## Notes

- **Active Development**: API may change between releases
- **PyTorch Compatibility**: Maintains numerical equivalence with torchvision
- **Transfer Learning**: All models support backbone extraction
- **Segmentation**: Includes dilated convolution support for dense prediction
- **Testing**: Comprehensive cross-framework validation ensures correctness