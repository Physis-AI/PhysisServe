"""Opt-in inference kernels shared by all scheduling strategies.

Importing this package does not import Triton or select an optimized backend.
Model adapters select kernels explicitly; unsupported inputs fail without fallback.
"""
