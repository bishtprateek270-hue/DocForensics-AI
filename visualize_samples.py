#!/usr/bin/env python3
"""
DocForensics AI — Visualization Entrypoint
Generates side-by-side visual inspection of documents and ground-truth masks.
"""
from src.preprocessing.visualize_dataset import visualize_dataset_samples


if __name__ == "__main__":
    visualize_dataset_samples(num_samples=6)
