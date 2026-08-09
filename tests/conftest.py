"""Pytest configuration and fixtures."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end (requires --run-e2e to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip slow and e2e tests by default unless flags are passed."""
    run_slow = config.getoption("--run-slow", default=False)
    run_e2e = config.getoption("--run-e2e", default=False)

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "e2e" in item.keywords and not run_e2e:
            item.add_marker(skip_e2e)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run end-to-end tests",
    )
