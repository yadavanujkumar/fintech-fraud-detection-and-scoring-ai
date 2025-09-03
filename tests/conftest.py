"""
Test configuration file.
"""

import pytest
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure test directories exist
    os.makedirs('test_data', exist_ok=True)
    os.makedirs('test_models', exist_ok=True)
    
    yield
    
    # Cleanup after tests
    import shutil
    if os.path.exists('test_data'):
        shutil.rmtree('test_data', ignore_errors=True)
    if os.path.exists('test_models'):
        shutil.rmtree('test_models', ignore_errors=True)