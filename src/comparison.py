def compare_embeddings(query_embedding, reference_embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Calculate cosine similarity between the query embedding and reference embeddings
    similarities = cosine_similarity(query_embedding.reshape(1, -1), reference_embeddings)
    
    # Return the similarities as a flattened array
    return similarities.flatten()