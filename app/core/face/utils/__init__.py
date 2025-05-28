"""Face recognition utilities module.

This module provides utilities utilities for face detection, recognition, and evaluation.
"""

from .create_embedding import (
    create_stable_features,
    save_features
)

from .face_utils import (
    extract_features,
    get_person_features,
    process_image,
    compare_embeddings
)

__all__ = [
    'create_stable_features',
    'save_features',
    'extract_features',
    'get_person_features',
    'process_image',
    'compare_embeddings'
]