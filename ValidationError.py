"""
This module defines custom exception classes for validation and GPU memory errors.
"""

class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class GPUOutOfMemoryError(Exception):
    """Exception raised for GPU out-of-memory errors."""
    pass
