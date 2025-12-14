"""
ML Application Package

This package contains the core machine learning application components
for distributed training and serving using Ray and MLflow on Kubernetes.
"""

__version__ = "1.0.0"
__author__ = "Alex Foley"

from .serve import ModelServer

__all__ = ["ModelServer"]
