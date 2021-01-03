from setuptools import setup, find_packages

version = '0.1.0'

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

setup(
    name='flaxvision',
    version=version,
    description='A selection of neural network models \
                ported from torchvision for JAX & Flax.',
    author='Roland Gavrilescu',
    author_email="gavrilescu.roland@gmail.com",
    url="https://github.com/rolandgvc/flaxvision",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
    })
