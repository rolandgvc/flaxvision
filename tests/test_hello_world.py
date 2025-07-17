"""
Tests for the hello_world.py example script.
"""

import sys
import os
import pytest
from io import StringIO
from unittest.mock import patch

# Add the examples directory to the path so we can import the hello_world module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from hello_world import hello_world


class TestHelloWorld:
    """Test class for hello_world functionality."""
    
    def test_hello_world_return_value(self):
        """Test that hello_world returns the expected string."""
        result = hello_world()
        assert result == "Hello, World!"
    
    def test_hello_world_output(self):
        """Test that hello_world prints the expected output."""
        # Capture stdout
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            hello_world()
        
        output = captured_output.getvalue()
        
        # Check that key messages are in the output
        assert "Hello, World!" in output
        assert "Welcome to FlaxVision" in output
        assert "Available FlaxVision models:" in output
        assert "FlaxVision is ready to use!" in output
        assert "Python version:" in output
        assert "Python executable:" in output
    
    def test_hello_world_output_contains_models(self):
        """Test that hello_world output contains model names."""
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            hello_world()
        
        output = captured_output.getvalue()
        
        # Check that some expected model names are in the output
        expected_models = ['vgg16', 'resnet18', 'resnet50', 'densenet121', 'inception_v3']
        for model in expected_models:
            assert model in output
    
    def test_hello_world_output_contains_description(self):
        """Test that hello_world output contains FlaxVision description."""
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            hello_world()
        
        output = captured_output.getvalue()
        
        # Check that the description is displayed
        assert "FlaxVision is a collection of neural network models" in output
        assert "ported from torchvision for JAX & Flax." in output
    
    def test_hello_world_python_info(self):
        """Test that hello_world displays Python environment info."""
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            hello_world()
        
        output = captured_output.getvalue()
        
        # Check that Python info is displayed
        assert "Python version:" in output
        assert "Python executable:" in output
        assert str(sys.version) in output
        assert sys.executable in output
    
    def test_hello_world_complete_flow(self):
        """Test the complete hello_world flow and output structure."""
        captured_output = StringIO()
        with patch('sys.stdout', captured_output):
            result = hello_world()
        
        output = captured_output.getvalue()
        
        # Test return value
        assert result == "Hello, World!"
        
        # Test output structure and content
        lines = output.strip().split('\n')
        assert len(lines) >= 10  # Should have multiple lines of output
        
        # Check that output flows logically
        assert "Hello, World!" in lines[0]
        assert "Welcome to FlaxVision" in lines[1]


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])