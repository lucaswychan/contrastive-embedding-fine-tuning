import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, device, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        temperature = np.log(temperature)
        self.temperature = nn.Parameter(torch.tensor(temperature).to(device))
        self.device = device

    def forward(self, query_embeddings, positive_embeddings, labels=None):
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(
            query_embeddings, positive_embeddings.t()
        ) / torch.exp(self.temperature)
        batch_size = query_embeddings.shape[0]

        if labels is not None:
            # Create mask where True indicates pairs with same label
            labels = labels.view(-1, 1)
            mask = (labels == labels.t()).float()

            # Remove self-contrast cases
            mask = mask.fill_diagonal_(0)

            # For numerical stability
            logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
            similarity_matrix = similarity_matrix - logits_max.detach()

            # Compute log_prob
            exp_logits = torch.exp(similarity_matrix)
            log_prob = similarity_matrix - torch.log(
                exp_logits.sum(dim=1, keepdim=True)
            )

            # Compute mean of log-likelihood over positive pairs
            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
            loss = -mean_log_prob_pos.mean()
        else:
            # Traditional InfoNCE without label information
            labels = torch.arange(batch_size, device=query_embeddings.device)
            loss = self.criterion(similarity_matrix, labels)

        return loss
