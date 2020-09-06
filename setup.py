from setuptools import setup, find_packages

version = '0.0.5'

install_requires = [
    'numpy>=1.12',
    'jax>=0.1.59',
    'flax==0.2.0'
]

tests_require = [
    'jaxlib',
    'pytest',
]

setup(
    name='flaxvision',
    version=version,
    description='A selection of neural network models \
                ported from torchvision for JAX & Flax.'
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
    author='Roland Gavrilescu'
    author_email="gavrilescu.roland@gmail.com"
    url="https://github.com/rolandgvc/flaxvision",
    install_requires=install_requires,
    packages=find_packages()
)
