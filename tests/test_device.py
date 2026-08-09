"""Tests for device detection utility."""


from gromit.utils.device import detect_device, resolve_device


def test_detect_device_returns_valid_device():
    """detect_device should return cuda, mps, or cpu."""
    device = detect_device()
    assert device in ("cuda", "mps", "cpu")


def test_resolve_device_auto():
    """resolve_device('auto') should call detect_device."""
    device = resolve_device("auto")
    assert device in ("cuda", "mps", "cpu")


def test_resolve_device_explicit():
    """resolve_device with explicit value should return that value."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"
