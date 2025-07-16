"""
Security test fixtures and utilities for the test suite.
Provides common fixtures, mock objects, and utilities for security testing.
"""
import pytest
import tempfile
import os
import gc
import time
import threading
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import jax.numpy as jnp
from jax import random
import flaxvision.models as models

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@pytest.fixture
def security_rng():
    """Provide a consistent random number generator for security tests."""
    return random.PRNGKey(42)


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    if not HAS_PSUTIL:
        pytest.skip("psutil not available")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    class MemoryMonitor:
        def __init__(self):
            self.initial_memory = initial_memory
            self.process = process
            self.peak_memory = initial_memory
            
        def get_current_usage(self):
            """Get current memory usage in MB."""
            return self.process.memory_info().rss / 1024 / 1024
            
        def get_increase(self):
            """Get memory increase since start in MB."""
            current = self.process.memory_info().rss
            return (current - self.initial_memory) / 1024 / 1024
            
        def update_peak(self):
            """Update peak memory usage."""
            current = self.process.memory_info().rss
            self.peak_memory = max(self.peak_memory, current)
            
        def get_peak_increase(self):
            """Get peak memory increase in MB."""
            return (self.peak_memory - self.initial_memory) / 1024 / 1024
    
    monitor = MemoryMonitor()
    yield monitor
    
    # Force garbage collection after test
    gc.collect()


@pytest.fixture
def mock_torch_hub():
    """Mock torch.hub.load_state_dict_from_url for security tests."""
    with patch('torch.hub.load_state_dict_from_url') as mock_load:
        # Default return value
        mock_load.return_value = {"test": "data"}
        yield mock_load


@pytest.fixture
def malicious_urls():
    """Provide a list of malicious URLs for testing."""
    return [
        "javascript:alert('xss')",
        "file:///etc/passwd",
        "ftp://malicious.com/file.pth",
        "http://localhost:22/ssh-exploit",
        "https://example.com/../../../etc/passwd",
        "https://example.com/model.pth?param=<script>alert('xss')</script>",
        "https://0.0.0.0/redirect",
        "https://127.0.0.1/internal",
        "https://169.254.169.254/metadata",  # AWS metadata
        "https://metadata.google.internal/computeMetadata/v1/",  # GCP metadata
        "ldap://evil.com/model.pth",
        "gopher://evil.com/model.pth",
        "data:text/plain;base64,SGVsbG8gV29ybGQ=",
    ]


@pytest.fixture
def valid_pytorch_urls():
    """Provide a list of valid PyTorch model URLs."""
    return [
        "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "https://download.pytorch.org/models/vgg16-397923af.pth",
        "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    ]


@pytest.fixture
def sample_model_and_params(security_rng):
    """Provide a sample model and parameters for testing."""
    model, params = models.resnet50(security_rng, pretrained=False)
    return model, params


@pytest.fixture
def sample_input():
    """Provide sample input data for model testing."""
    return jnp.ones((1, 224, 224, 3))


@pytest.fixture
def malicious_input_shapes():
    """Provide malicious input shapes for testing."""
    return [
        (-1, 224, 224, 3),  # Negative dimension
        (0, 224, 224, 3),   # Zero dimension
        (1, 99999, 99999, 3),  # Extremely large
        (1.5, 224, 224, 3),    # Float dimension
        ("malicious", 224, 224, 3),  # String dimension
        (None, 224, 224, 3),   # None dimension
    ]


@pytest.fixture
def resource_limits():
    """Provide resource limits for testing."""
    return {
        'memory_limit_mb': 1024,  # 1GB
        'timeout_seconds': 30,
        'max_cpu_percent': 90,
        'max_file_descriptors': 100,
    }


@pytest.fixture
def timeout_context():
    """Provide timeout context for long-running tests."""
    import signal
    
    class TimeoutContext:
        def __init__(self, timeout_seconds=30):
            self.timeout_seconds = timeout_seconds
            self.timed_out = False
            
        def __enter__(self):
            def timeout_handler(signum, frame):
                self.timed_out = True
                raise TimeoutError(f"Test timed out after {self.timeout_seconds} seconds")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            signal.alarm(0)  # Cancel timeout
    
    return TimeoutContext


@pytest.fixture
def network_error_simulator():
    """Simulate various network errors for testing."""
    import requests
    from urllib.error import URLError, HTTPError
    import ssl
    
    errors = {
        'timeout': requests.exceptions.Timeout("Request timeout"),
        'connection': requests.exceptions.ConnectionError("Connection failed"),
        'http_404': HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None),
        'http_500': HTTPError(url="", code=500, msg="Internal Server Error", hdrs=None, fp=None),
        'ssl_error': ssl.SSLError("SSL certificate error"),
        'url_error': URLError("Network unreachable"),
    }
    
    return errors


@pytest.fixture
def malicious_parameters():
    """Provide malicious parameter values for testing."""
    return {
        'nan_values': jnp.full((10, 10), jnp.nan),
        'inf_values': jnp.full((10, 10), jnp.inf),
        'very_large': jnp.full((10, 10), 1e20),
        'very_negative': jnp.full((10, 10), -1e20),
        'zero_values': jnp.zeros((10, 10)),
    }


@pytest.fixture
def thread_safety_tester():
    """Provide utilities for testing thread safety."""
    class ThreadSafetyTester:
        def __init__(self):
            self.results = []
            self.errors = []
            self.lock = threading.Lock()
            
        def add_result(self, result):
            with self.lock:
                self.results.append(result)
                
        def add_error(self, error):
            with self.lock:
                self.errors.append(error)
                
        def run_concurrent_test(self, test_func, num_threads=4, *args, **kwargs):
            """Run a test function concurrently in multiple threads."""
            def worker():
                try:
                    result = test_func(*args, **kwargs)
                    self.add_result(result)
                except Exception as e:
                    self.add_error(e)
            
            threads = []
            for _ in range(num_threads):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join(timeout=60)  # 60 second timeout
            
            return self.results, self.errors
    
    return ThreadSafetyTester()


@pytest.fixture
def security_test_data():
    """Provide comprehensive security test data."""
    return {
        'injection_strings': [
            "__import__('os').system('rm -rf /')",
            "eval('print(\"malicious\")')",
            "exec('import os; os.system(\"ls\")')",
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "$(rm -rf /)",
            "`rm -rf /`",
            "|| rm -rf /",
            "&& rm -rf /",
            "; rm -rf /",
        ],
        'path_traversal': [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc//passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd",
        ],
        'buffer_overflow': [
            "A" * 10000,
            "B" * 100000,
            "\x00" * 1000,
            "\xff" * 1000,
        ],
        'format_strings': [
            "%s%s%s%s%s%s%s%s%s%s%s%s",
            "%x%x%x%x%x%x%x%x%x%x%x%x",
            "%n%n%n%n%n%n%n%n%n%n%n%n",
        ],
    }


@pytest.fixture(autouse=True)
def security_test_setup():
    """Automatic setup for all security tests."""
    # Set up secure environment
    os.environ['PYTHONHASHSEED'] = '0'  # Reproducible hashing
    os.environ['JAX_ENABLE_X64'] = 'true'  # Enable 64-bit precision
    
    # Force garbage collection before each test
    gc.collect()
    
    yield
    
    # Cleanup after each test
    gc.collect()
    
    # Reset any global state that might have been modified
    import jax
    jax.clear_backends()


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.checkpoints = []
            
        def start(self):
            self.start_time = time.time()
            
        def checkpoint(self, name):
            if self.start_time is None:
                self.start()
            current_time = time.time()
            self.checkpoints.append((name, current_time - self.start_time))
            
        def finish(self):
            self.end_time = time.time()
            
        def get_total_time(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
            
        def get_checkpoints(self):
            return self.checkpoints
    
    return PerformanceMonitor()


# Pytest markers for security tests
def pytest_configure(config):
    """Configure pytest markers for security tests."""
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "download_security: mark test as download security-related"
    )
    config.addinivalue_line(
        "markers", "parameter_security: mark test as parameter security-related"
    )
    config.addinivalue_line(
        "markers", "resource_security: mark test as resource security-related"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "network: mark test as requiring network access"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: mark test as using significant memory"
    )


def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip network tests if no network access
    if item.get_closest_marker("network"):
        try:
            import requests
            requests.get("https://httpbin.org/status/200", timeout=5)
        except:
            pytest.skip("No network access available")


def pytest_runtest_teardown(item):
    """Teardown for individual test runs."""
    # Force garbage collection after each test
    gc.collect()
    
    # Reset any JAX state
    import jax
    jax.clear_backends()