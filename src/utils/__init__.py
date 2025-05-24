"""Face recognition utilities module.

This module provides common utilities for face detection, recognition, and evaluation.
"""

from .face_utils import (
    detect_faces,
    extract_features,
    compare_embeddings,
    is_match,
    process_image,
    get_target_name_from_dir
)

from .create_embedding import (
    create_stable_embedding,
    save_embedding
)