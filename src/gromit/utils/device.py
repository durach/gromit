"""Device detection utilities."""

import torch


def _test_cuda() -> bool:
    """Test if CUDA actually works (not just available).

    Returns:
        True if CUDA is functional, False otherwise
    """
    # A broken CUDA install fails at whichever layer breaks first: RuntimeError
    # from the allocator, OSError from a missing driver library, or a
    # torch-internal error. The probe's contract is to answer yes/no and never
    # propagate, so every failure mode means the same thing — not usable.
    try:
        # Create a small tensor on CUDA to verify driver compatibility
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        return True
    except Exception:  # noqa: BLE001 — any failure means CUDA is unusable; see above
        return False


def _test_mps() -> bool:
    """Test if MPS actually works (not just available).

    Returns:
        True if MPS is functional, False otherwise
    """
    # Same contract as _test_cuda: an unbuilt or unsupported MPS backend surfaces
    # as RuntimeError, NotImplementedError or a torch-internal error, and all of
    # them mean "not usable" rather than "crash the run".
    try:
        test_tensor = torch.zeros(1, device="mps")
        del test_tensor
        return True
    except Exception:  # noqa: BLE001 — any failure means MPS is unusable; see above
        return False


def detect_device() -> str:
    """Auto-detect the best available compute device.

    Tests actual functionality, not just availability, to catch
    driver version mismatches and other runtime issues.

    Returns:
        "cuda" if NVIDIA GPU functional
        "mps" if Apple Silicon functional
        "cpu" otherwise
    """
    if torch.cuda.is_available() and _test_cuda():
        return "cuda"
    elif torch.backends.mps.is_available() and _test_mps():
        return "mps"
    else:
        return "cpu"


def resolve_device(device: str) -> str:
    """Resolve device string, handling 'auto' detection.

    Args:
        device: Device string ("auto", "cuda", "mps", "cpu")

    Returns:
        Resolved device string
    """
    if device == "auto":
        return detect_device()
    return device
