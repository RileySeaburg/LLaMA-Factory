"""Rust-based code dataset converter."""

import os
import sys

# Add the compiled library directory to Python path
_lib_dir = os.path.dirname(os.path.abspath(__file__))
if _lib_dir not in sys.path:
    sys.path.append(_lib_dir)

try:
    from .rust_converter import convert_code_data, batch_convert_code_data
except ImportError as e:
    def _not_built(*args, **kwargs):
        raise ImportError(
            "Rust converter module not built. Please run 'cargo build --release' in "
            f"{_lib_dir} directory"
        ) from e
    
    convert_code_data = batch_convert_code_data = _not_built

__all__ = ["convert_code_data", "batch_convert_code_data"]
