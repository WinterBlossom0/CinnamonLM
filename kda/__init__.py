"""Kimi Delta Attention, with the sequential scan (scan_triton.py), the
gate/key preprocessing (gate_triton.py), and RMSNorm (rmsnorm_triton.py)
each fused into their own verified Triton kernel -- see attention.py's
docstring and the conversation history for what each targets and why.
"""
from .attention import KDA, inv_unit_lower

__all__ = ["KDA", "inv_unit_lower"]
