#!/usr/bin/env python3
"""
Validation script for the security testing framework.
This script validates the structure and basic functionality of the security tests.
"""

import os
import sys
import ast
import re

def validate_file_structure():
    """Validate that all required security test files exist."""
    required_files = [
        'tests/test_download_security.py',
        'tests/test_parameter_security.py', 
        'tests/test_resource_security.py',
        'tests/conftest.py',
        'pytest.ini'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required security test files exist")
        return True

def validate_test_structure(file_path):
    """Validate the structure of a test file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Parse the AST to analyze the structure
        tree = ast.parse(content)
        
        # Check for required imports
        has_unittest = any(
            isinstance(node, ast.Import) and 
            any(alias.name == 'unittest' for alias in node.names)
            for node in tree.body
        )
        
        has_pytest = any(
            isinstance(node, ast.Import) and 
            any(alias.name == 'pytest' for alias in node.names)
            for node in tree.body
        )
        
        # Check for test classes
        test_classes = [
            node for node in tree.body 
            if isinstance(node, ast.ClassDef) and 
            node.name.startswith('Test')
        ]
        
        # Check for test methods
        test_methods = []
        for class_node in test_classes:
            for method in class_node.body:
                if isinstance(method, ast.FunctionDef) and method.name.startswith('test_'):
                    test_methods.append(method.name)
        
        # Check for security markers
        has_security_markers = '@pytest.mark.security' in content
        
        return {
            'has_unittest': has_unittest,
            'has_pytest': has_pytest,
            'test_classes': len(test_classes),
            'test_methods': len(test_methods),
            'has_security_markers': has_security_markers,
            'method_names': test_methods[:5]  # First 5 methods
        }
        
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return None

def validate_security_coverage():
    """Validate that security tests cover the required areas."""
    test_files = {
        'tests/test_download_security.py': {
            'expected_tests': [
                'malicious_url', 'timeout', 'ssl', 'path_traversal', 
                'injection', 'checksum', 'redirect'
            ],
            'description': 'Download security tests'
        },
        'tests/test_parameter_security.py': {
            'expected_tests': [
                'input_shape', 'parameter_bounds', 'injection', 
                'nan_parameter', 'type_validation', 'serialization'
            ],
            'description': 'Parameter security tests'
        },
        'tests/test_resource_security.py': {
            'expected_tests': [
                'memory_exhaustion', 'timeout', 'cpu_usage', 
                'memory_leak', 'thread_safety', 'resource_cleanup'
            ],
            'description': 'Resource security tests'
        }
    }
    
    coverage_results = {}
    
    for file_path, expected in test_files.items():
        if not os.path.exists(file_path):
            coverage_results[file_path] = {'status': 'missing', 'coverage': 0}
            continue
            
        with open(file_path, 'r') as f:
            content = f.read().lower()
            
        found_tests = []
        for test_pattern in expected['expected_tests']:
            if test_pattern in content:
                found_tests.append(test_pattern)
        
        coverage_percent = (len(found_tests) / len(expected['expected_tests'])) * 100
        coverage_results[file_path] = {
            'status': 'exists',
            'coverage': coverage_percent,
            'found_tests': found_tests,
            'description': expected['description']
        }
    
    return coverage_results

def validate_pytest_configuration():
    """Validate pytest configuration."""
    if not os.path.exists('pytest.ini'):
        print("❌ pytest.ini not found")
        return False
    
    with open('pytest.ini', 'r') as f:
        content = f.read()
    
    required_markers = [
        'security:', 'download_security:', 'parameter_security:', 
        'resource_security:', 'slow:', 'network:', 'memory_intensive:'
    ]
    
    missing_markers = []
    for marker in required_markers:
        if marker not in content:
            missing_markers.append(marker)
    
    if missing_markers:
        print(f"❌ Missing pytest markers: {missing_markers}")
        return False
    else:
        print("✅ All required pytest markers configured")
        return True

def validate_conftest_fixtures():
    """Validate conftest.py fixtures."""
    if not os.path.exists('tests/conftest.py'):
        print("❌ conftest.py not found")
        return False
    
    with open('tests/conftest.py', 'r') as f:
        content = f.read()
    
    expected_fixtures = [
        'security_rng', 'temp_directory', 'malicious_urls', 
        'valid_pytorch_urls', 'sample_model_and_params', 'resource_limits'
    ]
    
    found_fixtures = []
    for fixture in expected_fixtures:
        if f'def {fixture}(' in content:
            found_fixtures.append(fixture)
    
    coverage_percent = (len(found_fixtures) / len(expected_fixtures)) * 100
    
    print(f"✅ Conftest fixtures coverage: {coverage_percent:.1f}% ({len(found_fixtures)}/{len(expected_fixtures)})")
    
    if coverage_percent < 80:
        print(f"⚠️  Low fixture coverage. Missing: {set(expected_fixtures) - set(found_fixtures)}")
    
    return coverage_percent >= 80

def main():
    """Main validation function."""
    print("🔒 Security Testing Framework Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Validate file structure
    if not validate_file_structure():
        all_passed = False
    
    # Validate pytest configuration
    if not validate_pytest_configuration():
        all_passed = False
    
    # Validate conftest fixtures
    if not validate_conftest_fixtures():
        all_passed = False
    
    # Validate test structure
    test_files = [
        'tests/test_download_security.py',
        'tests/test_parameter_security.py', 
        'tests/test_resource_security.py'
    ]
    
    print("\n📊 Test File Analysis:")
    for file_path in test_files:
        result = validate_test_structure(file_path)
        if result:
            print(f"\n{file_path}:")
            print(f"  - Test classes: {result['test_classes']}")
            print(f"  - Test methods: {result['test_methods']}")
            print(f"  - Has pytest markers: {'✅' if result['has_security_markers'] else '❌'}")
            print(f"  - Sample methods: {result['method_names']}")
        else:
            all_passed = False
    
    # Validate security coverage
    print("\n🛡️  Security Coverage Analysis:")
    coverage_results = validate_security_coverage()
    
    for file_path, result in coverage_results.items():
        if result['status'] == 'exists':
            print(f"\n{result['description']} ({file_path}):")
            print(f"  - Coverage: {result['coverage']:.1f}%")
            print(f"  - Found tests: {result['found_tests']}")
            if result['coverage'] < 80:
                print(f"  - ⚠️  Low coverage")
                all_passed = False
            else:
                print(f"  - ✅ Good coverage")
        else:
            print(f"\n{file_path}: ❌ Missing")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 Security Testing Framework Validation PASSED")
        print("All security tests are properly structured and have good coverage.")
    else:
        print("❌ Security Testing Framework Validation FAILED")
        print("Some issues need to be addressed.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)