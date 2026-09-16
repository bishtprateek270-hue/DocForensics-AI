"""
DocForensics AI — Forensics Module Package (Phase 7)
Modular forensic feature extractors: SRM Filters, ELA, and Local Noise Residuals.
"""

from src.forensics.srm_filters import SRMExtractor, get_srm_extractor, get_srm_kernel_weights
from src.forensics.ela_extractor import ELAExtractor, compute_ela_numpy
from src.forensics.noise_extractor import LocalNoiseExtractor

__all__ = [
    "SRMExtractor",
    "get_srm_extractor",
    "get_srm_kernel_weights",
    "ELAExtractor",
    "compute_ela_numpy",
    "LocalNoiseExtractor",
]
