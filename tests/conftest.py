"""Shared test fixtures for the perception pipeline tests."""

import sys
import os

# Add parent dir to path so we can import perception modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add tests dir to path for test_helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import pytest
from config import DEFAULT_CONFIG


@pytest.fixture
def default_config():
    """Return a fresh copy of the default config."""
    return copy.deepcopy(DEFAULT_CONFIG)
