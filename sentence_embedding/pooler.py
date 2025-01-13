import torch
import torch.nn.functional as F


def pool_embeddings(
    embeddings: torch.Tensor, pooling_type: str = "mean"
) -> torch.Tensor:
    """
    Pool multiple sentence embeddings into a single embedding vector.

    Args:
        embeddings: numpy array of shape (num_sentences, embedding_dim)
        pooling_method: str, one of ['mean', 'max']

    Returns:
        pooled_embedding: numpy array of shape (embedding_dim,)
    """
    if len(embeddings.shape) != 2:
        raise ValueError(
            "Input embeddings must be 2-dimensional with shape (num_sentences, embedding_dim)"
        )

    if pooling_type == "mean":
        # Average pooling across sentences
        return torch.mean(embeddings, axis=0)

    elif pooling_type == "max":
        # Max pooling across sentences
        return torch.max(embeddings, axis=0)
    else:
        raise ValueError("pooling_type must be one of ['mean', 'max']")
