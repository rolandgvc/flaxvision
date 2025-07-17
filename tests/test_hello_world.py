"""
Tests for the hello world script
"""

import pytest
import sys
import os
from io import StringIO

# Add the parent directory to the path so we can import hello_world
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hello_world import hello_world, main


class TestHelloWorld:
    """Test class for hello world functionality."""

    def test_hello_world_returns_correct_message(self):
        """Test that hello_world returns the correct message."""
        result = hello_world()
        assert result == "Hello, World!"

    def test_hello_world_return_type(self):
        """Test that hello_world returns a string."""
        result = hello_world()
        assert isinstance(result, str)

    def test_main_function_prints_hello_world(self, capsys):
        """Test that main function prints hello world message."""
        main()
        captured = capsys.readouterr()
        assert captured.out.strip() == "Hello, World!"

    def test_main_function_no_error(self):
        """Test that main function runs without errors."""
        try:
            main()
        except Exception as e:
            pytest.fail(f"main() raised an exception: {e}")