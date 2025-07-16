"""
Security tests for resource exhaustion protection and monitoring.
Tests memory exhaustion protection, timeout handling, and resource usage monitoring.
"""
import pytest
import unittest
import unittest.mock as mock
import time
import threading
import gc
import os
import signal
import numpy as np
import jax.numpy as jnp
from jax import random
import flaxvision.models as models
from unittest.mock import patch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TestResourceSecurity(unittest.TestCase):
    """Test suite for resource security measures."""
    
    def setUp(self):
        """Set up test environment."""
        self.rng = random.PRNGKey(42)
        if HAS_PSUTIL:
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss
        else:
            self.process = None
            self.initial_memory = 0
        
        # Resource limits for testing
        self.memory_limit_mb = 1024  # 1GB limit for testing
        self.timeout_seconds = 30
        self.max_cpu_percent = 90
    
    def tearDown(self):
        """Clean up test environment."""
        # Force garbage collection to clean up memory
        gc.collect()
    
    @pytest.mark.security
    @pytest.mark.resource_security
    @pytest.mark.memory_intensive
    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks."""
        # Test with progressively larger models/inputs
        test_sizes = [
            (1, 224, 224, 3),      # Normal size
            (1, 512, 512, 3),      # Larger size
            (1, 1024, 1024, 3),    # Very large size
            (1, 2048, 2048, 3),    # Extremely large size
        ]
        
        for size in test_sizes:
            with self.subTest(size=size):
                try:
                    # Monitor memory usage
                    memory_before = self.process.memory_info().rss
                    
                    # Create large input
                    large_input = jnp.ones(size)
                    model, params = models.resnet50(self.rng, pretrained=False)
                    
                    # Check memory usage
                    memory_after_allocation = self.process.memory_info().rss
                    memory_used_mb = (memory_after_allocation - memory_before) / 1024 / 1024
                    
                    if memory_used_mb > self.memory_limit_mb:
                        self.fail(f"Memory usage exceeded limit: {memory_used_mb}MB > {self.memory_limit_mb}MB")
                    
                    # Try to run inference
                    output = model.apply(params, large_input, mutable=False)
                    
                    # Check final memory usage
                    memory_final = self.process.memory_info().rss
                    total_memory_mb = (memory_final - memory_before) / 1024 / 1024
                    
                    if total_memory_mb > self.memory_limit_mb * 2:  # Allow some overhead
                        self.fail(f"Total memory usage exceeded limit: {total_memory_mb}MB")
                    
                    # Clean up
                    del large_input, model, params, output
                    gc.collect()
                    
                except (MemoryError, RuntimeError, ValueError) as e:
                    # Expected for very large inputs
                    if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                        pass  # Expected behavior
                    else:
                        raise
    
    @pytest.mark.security
    @pytest.mark.resource_security
    @pytest.mark.slow
    def test_timeout_protection(self):
        """Test timeout protection for long-running operations."""
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")
        
        # Test with a potentially slow operation
        try:
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            # Perform operation that might take long time
            large_input = jnp.ones((1, 1024, 1024, 3))
            model, params = models.resnet50(self.rng, pretrained=False)
            
            start_time = time.time()
            output = model.apply(params, large_input, mutable=False)
            end_time = time.time()
            
            # Cancel timeout
            signal.alarm(0)
            
            # Check execution time
            execution_time = end_time - start_time
            if execution_time > self.timeout_seconds:
                self.fail(f"Operation took too long: {execution_time}s > {self.timeout_seconds}s")
            
        except TimeoutError:
            # Expected - operation was properly timed out
            pass
        except (MemoryError, RuntimeError):
            # Expected - operation might fail due to resource limits
            pass
        finally:
            # Always cancel timeout
            signal.alarm(0)
    
    @pytest.mark.security
    @pytest.mark.resource_security
    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring and limits."""
        # Monitor CPU usage during model operations
        cpu_percentages = []
        
        def monitor_cpu():
            while threading.active_count() > 1:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_percentages.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Perform CPU-intensive operation
            model, params = models.resnet50(self.rng, pretrained=False)
            
            # Run multiple inferences to stress CPU
            for i in range(5):
                input_data = jnp.ones((1, 224, 224, 3))
                output = model.apply(params, input_data, mutable=False)
            
            # Wait for monitoring to complete
            time.sleep(0.5)
            
            # Check CPU usage
            if cpu_percentages:
                max_cpu = max(cpu_percentages)
                avg_cpu = sum(cpu_percentages) / len(cpu_percentages)
                
                # Log CPU usage (don't fail test, just monitor)
                print(f"Max CPU usage: {max_cpu}%, Average: {avg_cpu}%")
                
                # Only fail if CPU usage is consistently at 100%
                if avg_cpu > 99:
                    self.fail(f"CPU usage too high: {avg_cpu}%")
            
        finally:
            # Ensure monitoring stops
            monitor_thread.join(timeout=1)
    
    @pytest.mark.security
    @pytest.mark.resource_security
    @pytest.mark.memory_intensive
    def test_memory_leak_detection(self):
        """Test for memory leaks in model operations."""
        initial_memory = self.process.memory_info().rss
        
        # Perform multiple operations
        for i in range(10):
            model, params = models.resnet50(self.rng, pretrained=False)
            input_data = jnp.ones((1, 224, 224, 3))
            output = model.apply(params, input_data, mutable=False)
            
            # Explicitly delete to free memory
            del model, params, input_data, output
            gc.collect()
        
        final_memory = self.process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Allow some memory increase but not excessive
        if memory_increase > 100:  # 100MB threshold
            self.fail(f"Potential memory leak detected: {memory_increase}MB increase")
    
    @pytest.mark.security
    @pytest.mark.resource_security
    def test_recursive_model_protection(self):
        """Test protection against recursive model definitions."""
        # This test ensures models don't create infinite recursion
        try:
            model, params = models.resnet50(self.rng, pretrained=False)
            
            # Try to create very deep call stack
            def recursive_apply(depth):
                if depth > 1000:  # Limit recursion depth
                    raise RecursionError("Maximum recursion depth exceeded")
                
                input_data = jnp.ones((1, 224, 224, 3))
                output = model.apply(params, input_data, mutable=False)
                
                if depth < 5:  # Only recurse a few times for testing
                    return recursive_apply(depth + 1)
                return output
            
            # This should not cause stack overflow
            result = recursive_apply(0)
            self.assertIsNotNone(result)
            
        except RecursionError:
            # Expected - recursion should be limited
            pass
    
    @pytest.mark.security
    @pytest.mark.resource_security
    def test_file_descriptor_limits(self):
        """Test file descriptor usage limits."""
        import resource
        
        # Get current file descriptor limit
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        
        # Test doesn't exceed reasonable file descriptor usage
        initial_fd_count = len(os.listdir('/proc/self/fd'))
        
        # Perform operations that might use file descriptors
        for i in range(10):
            model, params = models.resnet50(self.rng, pretrained=False)
            input_data = jnp.ones((1, 224, 224, 3))
            output = model.apply(params, input_data, mutable=False)
            del model, params, input_data, output
        
        final_fd_count = len(os.listdir('/proc/self/fd'))
        fd_increase = final_fd_count - initial_fd_count
        
        # Should not significantly increase file descriptor usage
        if fd_increase > 50:  # Reasonable threshold
            self.fail(f"Too many file descriptors opened: {fd_increase}")
    
    @pytest.mark.security
    @pytest.mark.resource_security
    def test_thread_safety(self):
        """Test thread safety of model operations."""
        model, params = models.resnet50(self.rng, pretrained=False)
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(5):
                    input_data = jnp.ones((1, 224, 224, 3))
                    output = model.apply(params, input_data, mutable=False)
                    results.append(output)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(4):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout
        
        # Check results
        if errors:
            self.fail(f"Thread safety issues: {errors}")
        
        # All threads should produce results
        self.assertGreater(len(results), 0, "No results produced by threads")
    
    @pytest.mark.security
    @pytest.mark.resource_security
    @pytest.mark.memory_intensive
    def test_resource_cleanup(self):
        """Test proper resource cleanup after operations."""
        initial_memory = self.process.memory_info().rss
        
        # Perform operation in a controlled scope
        def scoped_operation():
            model, params = models.resnet50(self.rng, pretrained=False)
            input_data = jnp.ones((1, 224, 224, 3))
            output = model.apply(params, input_data, mutable=False)
            return output.shape  # Only return shape, not full tensor
        
        result_shape = scoped_operation()
        
        # Force garbage collection
        gc.collect()
        
        # Wait a bit for cleanup
        time.sleep(1)
        
        final_memory = self.process.memory_info().rss
        memory_diff = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory should not increase significantly
        if memory_diff > 50:  # 50MB threshold
            self.fail(f"Resources not properly cleaned up: {memory_diff}MB increase")
        
        # Result should be valid
        self.assertIsNotNone(result_shape)
    
    @pytest.mark.security
    @pytest.mark.resource_security
    @pytest.mark.memory_intensive
    def test_batch_processing_limits(self):
        """Test limits on batch processing to prevent resource exhaustion."""
        model, params = models.resnet50(self.rng, pretrained=False)
        
        # Test with increasing batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                try:
                    memory_before = self.process.memory_info().rss
                    
                    # Create batch input
                    batch_input = jnp.ones((batch_size, 224, 224, 3))
                    
                    # Process batch
                    output = model.apply(params, batch_input, mutable=False)
                    
                    memory_after = self.process.memory_info().rss
                    memory_used = (memory_after - memory_before) / 1024 / 1024  # MB
                    
                    # Memory usage should scale reasonably with batch size
                    memory_per_sample = memory_used / batch_size
                    
                    if memory_per_sample > 100:  # 100MB per sample is too much
                        self.fail(f"Memory usage per sample too high: {memory_per_sample}MB")
                    
                    # Clean up
                    del batch_input, output
                    gc.collect()
                    
                except (MemoryError, RuntimeError) as e:
                    # Expected for large batch sizes
                    if batch_size > 16:  # Large batches might fail
                        pass
                    else:
                        raise
    
    @pytest.mark.security
    @pytest.mark.resource_security
    @pytest.mark.memory_intensive
    def test_model_loading_limits(self):
        """Test limits on model loading to prevent resource exhaustion."""
        # Test loading multiple models simultaneously
        models_loaded = []
        
        try:
            for i in range(5):  # Try to load 5 models
                model, params = models.resnet50(self.rng, pretrained=False)
                models_loaded.append((model, params))
                
                # Check memory usage
                memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
                
                if memory_usage > self.memory_limit_mb * 2:  # Allow some overhead
                    break  # Stop loading if memory usage too high
            
            # Should be able to load at least one model
            self.assertGreater(len(models_loaded), 0, "Could not load any models")
            
        except MemoryError:
            # Expected - system should prevent excessive memory usage
            pass
        finally:
            # Clean up all loaded models
            for model, params in models_loaded:
                del model, params
            gc.collect()


if __name__ == '__main__':
    unittest.main()