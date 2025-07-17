#!/usr/bin/env python3
"""
Hello World Example for FlaxVision

This is a simple hello world script that demonstrates basic usage of the FlaxVision library.
"""

import sys
import os

# Add the flaxvision package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def hello_world():
    """
    A simple hello world function that demonstrates FlaxVision usage.
    """
    print("Hello, World!")
    print("Welcome to FlaxVision - JAX/Flax Computer Vision Models")
    
    # Show basic information about FlaxVision
    print("\nFlaxVision is a collection of neural network models")
    print("ported from torchvision for JAX & Flax.")
    
    # Show available models
    print("\nAvailable FlaxVision models:")
    available_models = ['vgg16', 'resnet18', 'resnet50', 'densenet121', 'inception_v3']
    for model_name in available_models:
        print(f"  - {model_name}")
    
    # Show basic Python environment info
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    print("\nFlaxVision is ready to use!")
    return "Hello, World!"


if __name__ == "__main__":
    result = hello_world()
    print(f"\nScript completed successfully: {result}")