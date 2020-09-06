from setuptools import setup, find_packages

version = '0.0.5'

install_requires = [
    'numpy',
    'jax>=0.1.59',
    'flax==0.2.0',
    'torch>=1.3.0',
]

tests_require = [
    'jaxlib',
    'pytest',
]

setup(
    name='flaxvision',
    version=version,
    description='A selection of neural network models \
                ported from torchvision for JAX & Flax.',
    author='Roland Gavrilescu',
    author_email="gavrilescu.roland@gmail.com",
    url="https://github.com/rolandgvc/flaxvision",
    packages=find_packages()
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
        },
)
