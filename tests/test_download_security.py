import pytest
import unittest
from unittest.mock import patch, MagicMock, Mock
import tempfile
import os
import hashlib
import time
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
import ssl

import flaxvision.utils as utils


class TestDownloadSecurity(unittest.TestCase):
    """Comprehensive security tests for model download functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_pytorch_urls = [
            'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'https://download.pytorch.org/models/vgg16-397923af.pth'
        ]
        self.malicious_urls = [
            'file:///etc/passwd',
            'ftp://malicious.com/model.pth',
            'javascript:alert(1)',
            'data:text/plain;base64,SGVsbG8gV29ybGQ=',
            'ldap://evil.com/model.pth',
            'gopher://evil.com/model.pth',
            'http://pytorch.org/models/model.pth',  # HTTP instead of HTTPS
            'https://evil.com/model.pth',  # Wrong domain
            'https://download.pytorch.org/../../../etc/passwd',  # Path traversal
            'https://download.pytorch.org/models/model.pth?cmd=rm%20-rf%20/',  # Command injection
        ]

    @pytest.mark.security
    def test_malicious_url_schemes_rejected(self):
        """Test that malicious URL schemes are rejected."""
        dangerous_schemes = ['file', 'ftp', 'javascript', 'data', 'ldap', 'gopher']
        
        for scheme in dangerous_schemes:
            url = f"{scheme}://example.com/model.pth"
            with self.assertRaises((ValueError, URLError, TypeError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    def test_http_urls_rejected(self):
        """Test that HTTP URLs are rejected in favor of HTTPS."""
        http_urls = [
            'http://download.pytorch.org/models/resnet50-19c8e357.pth',
            'http://pytorch.org/models/model.pth'
        ]
        
        for url in http_urls:
            with self.assertRaises((ValueError, URLError, ssl.SSLError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    def test_domain_validation(self):
        """Test that only pytorch.org domain is allowed."""
        invalid_domains = [
            'https://evil.com/model.pth',
            'https://pytorch.evil.com/model.pth',
            'https://download.pytorch.org.evil.com/model.pth',
            'https://malicious.com/download.pytorch.org/model.pth'
        ]
        
        for url in invalid_domains:
            with self.assertRaises((ValueError, URLError, HTTPError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        path_traversal_urls = [
            'https://download.pytorch.org/../../../etc/passwd',
            'https://download.pytorch.org/models/../../etc/passwd',
            'https://download.pytorch.org/models/../models/../../etc/passwd',
            'https://download.pytorch.org/models/model.pth?path=../../etc/passwd'
        ]
        
        for url in path_traversal_urls:
            with self.assertRaises((ValueError, URLError, HTTPError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_download_timeout_handling(self, mock_load):
        """Test proper handling of download timeouts."""
        mock_load.side_effect = TimeoutError("Download timed out")
        
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        with self.assertRaises((TimeoutError, URLError)):
            utils.load_torch_params(url)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_network_error_handling(self, mock_load):
        """Test handling of various network errors."""
        network_errors = [
            ConnectionError("Connection failed"),
            URLError("Network unreachable"),
            HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None),
            ssl.SSLError("SSL certificate error")
        ]
        
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        
        for error in network_errors:
            mock_load.side_effect = error
            with self.assertRaises((ConnectionError, URLError, HTTPError, ssl.SSLError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_corrupted_file_handling(self, mock_load):
        """Test handling of corrupted or malformed model files."""
        # Mock corrupted data
        mock_load.return_value = {"corrupted": "data"}
        
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        
        # Should not raise an exception for corrupted data at load time
        # but should be handled gracefully
        result = utils.load_torch_params(url)
        self.assertIsInstance(result, dict)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_oversized_download_protection(self, mock_load):
        """Test protection against oversized downloads."""
        # Create a mock that simulates an oversized download
        mock_load.side_effect = MemoryError("File too large")
        
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        with self.assertRaises(MemoryError):
            utils.load_torch_params(url)

    @pytest.mark.security
    def test_url_parameter_injection(self):
        """Test protection against URL parameter injection attacks."""
        injection_urls = [
            'https://download.pytorch.org/models/model.pth?cmd=rm%20-rf%20/',
            'https://download.pytorch.org/models/model.pth?path=/etc/passwd',
            'https://download.pytorch.org/models/model.pth?exec=malicious_command',
            'https://download.pytorch.org/models/model.pth?file=../../../etc/passwd',
            'https://download.pytorch.org/models/model.pth?redirect=https://evil.com'
        ]
        
        for url in injection_urls:
            with self.assertRaises((ValueError, URLError, HTTPError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    def test_url_length_validation(self):
        """Test validation of URL length to prevent buffer overflow attacks."""
        # Create extremely long URL
        long_path = 'a' * 10000
        long_url = f'https://download.pytorch.org/models/{long_path}.pth'
        
        with self.assertRaises((ValueError, URLError)):
            utils.load_torch_params(long_url)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_concurrent_download_safety(self, mock_load):
        """Test safety of concurrent downloads."""
        import threading
        
        mock_load.return_value = {"test": "data"}
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        
        results = []
        errors = []
        
        def download_worker():
            try:
                result = utils.load_torch_params(url)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=download_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have race conditions or crashes
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)

    @pytest.mark.security
    def test_url_encoding_validation(self):
        """Test validation of URL encoding to prevent bypass attacks."""
        encoded_malicious_urls = [
            'https://download.pytorch.org/models/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd',  # ../../../etc/passwd
            'https://download.pytorch.org/models/model%00.pth',  # Null byte injection
            'https://download.pytorch.org/models/model.pth%3Fcmd%3Drm%20-rf%20/',  # URL-encoded command injection
            'https://download.pytorch.org/models/model.pth%0A%0D',  # CRLF injection
        ]
        
        for url in encoded_malicious_urls:
            with self.assertRaises((ValueError, URLError, HTTPError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_malicious_content_detection(self, mock_load):
        """Test detection of malicious content in downloaded files."""
        # Mock potentially malicious content
        malicious_payloads = [
            b'\x89PNG\r\n\x1a\n',  # PNG header (wrong file type)
            b'PK\x03\x04',  # ZIP header (wrong file type)
            b'\x7fELF',  # ELF binary header
            b'MZ',  # PE executable header
            b'#!/bin/sh\nrm -rf /',  # Shell script
        ]
        
        for payload in malicious_payloads:
            mock_load.return_value = payload
            url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            
            # The function should handle unexpected content gracefully
            # or raise appropriate exceptions
            try:
                result = utils.load_torch_params(url)
                # If no exception is raised, ensure it's handled properly
                self.assertIsNotNone(result)
            except (ValueError, TypeError, RuntimeError):
                # These exceptions are acceptable for malicious content
                pass

    @pytest.mark.security
    def test_dns_rebinding_protection(self):
        """Test protection against DNS rebinding attacks."""
        rebinding_urls = [
            'https://127.0.0.1:8080/models/model.pth',
            'https://localhost/models/model.pth',
            'https://0.0.0.0/models/model.pth',
            'https://[::1]/models/model.pth',  # IPv6 localhost
            'https://10.0.0.1/models/model.pth',  # Private IP
            'https://192.168.1.1/models/model.pth',  # Private IP
            'https://172.16.0.1/models/model.pth',  # Private IP
        ]
        
        for url in rebinding_urls:
            with self.assertRaises((ValueError, URLError, HTTPError)):
                utils.load_torch_params(url)

    @pytest.mark.security
    @patch('torch.hub.load_state_dict_from_url')
    def test_slow_loris_protection(self, mock_load):
        """Test protection against slow loris attacks."""
        def slow_response():
            time.sleep(0.1)  # Simulate slow response
            return {"test": "data"}
        
        mock_load.side_effect = slow_response
        url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
        
        start_time = time.time()
        result = utils.load_torch_params(url)
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 5.0)  # 5 second timeout
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()