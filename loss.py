import torch.nn as nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, terminology_labels):
        """
        Compute contrastive loss for terminology-based sentence embeddings

        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            terminology_labels: Tensor of shape [batch_size] containing terminology IDs
            temperature: Temperature parameter to scale the similarity scores
        """

        # Compute similarity matrix
        # Normalize embeddings for cosine similarity
        normalized_embeddings = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)

        # Scale similarities by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Create mask for positive pairs (same terminology)
        labels_matrix = terminology_labels.unsqueeze(0) == terminology_labels.unsqueeze(
            1
        )
        labels_matrix = labels_matrix.float()

        # Remove self-contrast
        mask_self = torch.eye(labels_matrix.shape[0], device=labels_matrix.device)
        labels_matrix = labels_matrix - mask_self

        # Compute log_prob
        exp_sim = torch.exp(similarity_matrix)

        # Create mask to exclude self-similarity
        mask_self_neg = 1 - mask_self

        # Calculate denominator (sum of all exp similarities except self)
        denominator = (exp_sim * mask_self_neg).sum(dim=1)

        # Calculate positive pairs
        positive_pairs = similarity_matrix[labels_matrix.bool()].reshape(
            embeddings.shape[0], -1
        )

        # If a terminology has only one word, skip it in loss calculation
        valid_samples = positive_pairs.shape[1] > 0

        if valid_samples.any():
            log_prob = positive_pairs - torch.log(denominator).unsqueeze(1)
            loss = -log_prob.mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        return loss
