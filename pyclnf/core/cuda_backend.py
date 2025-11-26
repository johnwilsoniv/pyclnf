"""
CUDA Backend Detection and Selection for pyCLNF.

Provides automatic detection of CUDA availability and device selection.
"""

import os
from typing import Optional

# Global state for backend selection
_forced_backend: Optional[str] = None
_cuda_available: Optional[bool] = None


class ComputeBackend:
    """Compute backend constants and detection."""

    CUDA = "cuda"
    CPU = "cpu"

    @classmethod
    def is_cuda_available(cls) -> bool:
        """
        Check if CUDA is available via PyTorch.

        Returns:
            True if CUDA is available and working, False otherwise.
        """
        global _cuda_available

        if _cuda_available is not None:
            return _cuda_available

        try:
            import torch
            _cuda_available = torch.cuda.is_available()
            if _cuda_available:
                # Verify we can actually use CUDA
                try:
                    torch.zeros(1, device='cuda')
                except Exception:
                    _cuda_available = False
        except ImportError:
            _cuda_available = False

        return _cuda_available

    @classmethod
    def get_device_name(cls) -> Optional[str]:
        """
        Get the name of the CUDA device if available.

        Returns:
            Device name string or None if CUDA not available.
        """
        if not cls.is_cuda_available():
            return None

        try:
            import torch
            return torch.cuda.get_device_name(0)
        except Exception:
            return None

    @classmethod
    def get(cls) -> str:
        """
        Get the current compute backend.

        Respects forced backend setting, otherwise auto-detects.

        Returns:
            "cuda" if CUDA available and not forced to CPU, else "cpu"
        """
        global _forced_backend

        if _forced_backend is not None:
            return _forced_backend

        # Check environment variable
        env_backend = os.environ.get("PYCLNF_BACKEND", "").lower()
        if env_backend in ("cpu", "cuda"):
            return env_backend if env_backend == "cpu" or cls.is_cuda_available() else "cpu"

        return cls.CUDA if cls.is_cuda_available() else cls.CPU

    @classmethod
    def set(cls, backend: str) -> None:
        """
        Force a specific backend.

        Args:
            backend: "cuda", "cpu", or None (auto-detect)

        Raises:
            ValueError: If invalid backend specified
            RuntimeError: If CUDA requested but not available
        """
        global _forced_backend

        if backend is None:
            _forced_backend = None
            return

        backend = backend.lower()
        if backend not in (cls.CUDA, cls.CPU):
            raise ValueError(f"Invalid backend: {backend}. Must be 'cuda', 'cpu', or None.")

        if backend == cls.CUDA and not cls.is_cuda_available():
            raise RuntimeError(
                "CUDA backend requested but not available. "
                "Install PyTorch with CUDA support: "
                "pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )

        _forced_backend = backend

    @classmethod
    def reset(cls) -> None:
        """Reset to auto-detection mode."""
        global _forced_backend, _cuda_available
        _forced_backend = None
        _cuda_available = None


def get_backend() -> str:
    """Convenience function to get current backend."""
    return ComputeBackend.get()


def set_backend(backend: str) -> None:
    """Convenience function to set backend."""
    ComputeBackend.set(backend)


def is_cuda_available() -> bool:
    """Convenience function to check CUDA availability."""
    return ComputeBackend.is_cuda_available()
