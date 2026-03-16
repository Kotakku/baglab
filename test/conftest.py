"""Shared fixtures for baglab tests."""

import shutil
from pathlib import Path

import pytest

import baglab


TEST_DIR = Path(__file__).parent
TEST_BAG_PATH = TEST_DIR / "test_bag"


@pytest.fixture(scope="session")
def test_bag_path():
    """Path to the test rosbag directory."""
    if not TEST_BAG_PATH.exists():
        pytest.skip(
            f"Test bag not found at {TEST_BAG_PATH}. "
            "Run ./generate_test_bag.sh first."
        )
    return TEST_BAG_PATH


@pytest.fixture()
def bag(test_bag_path):
    """Opened Bag handle, closed after test."""
    b = baglab.load(test_bag_path)
    yield b
    b.close()


@pytest.fixture()
def writable_bag(test_bag_path, tmp_path):
    """Bag loaded from a writable copy so disk cache can be created."""
    dst = tmp_path / "test_bag"
    shutil.copytree(test_bag_path, dst)
    b = baglab.load(dst)
    yield b
    b.close()
