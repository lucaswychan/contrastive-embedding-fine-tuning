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
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_inputs, positive_inputs, labels=None):
        # Normalize embeddings
        query_embeddings = query_inputs.get("sentence_embedding", None)
        positive_embeddings = positive_inputs.get("sentence_embedding", None)
        
        device = query_embeddings.device
        
        contrast_count = query_embeddings.shape[1]
        
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.div(torch.matmul(
            query_embeddings, positive_embeddings.t()
        ), torch.exp(self.temperature) )
        batch_size = query_embeddings.shape[0]

        if labels is not None:
            # Create mask where True indicates pairs with same label
            labels = labels.contiguous().view(-1, 1)
            mask = (labels == labels.t()).float().to(device)

            # For numerical stability
            logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
            logits = similarity_matrix - logits_max.detach()
            
            # tile mask
            mask = mask.repeat(contrast_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
                0,
            )
            
            mask = mask * logits_mask

            # Compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            
            log_prob = logits - torch.log(
                exp_logits.sum(dim=1, keepdim=True)
            )
            
            mask_pos_pairs = mask.sum(1)
            mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

            loss = -mean_log_prob_pos
            loss = loss.view(contrast_count, batch_size).mean()
        else:
            # Traditional InfoNCE without label information
            labels = torch.arange(batch_size, device=query_embeddings.device)
            loss = self.criterion(similarity_matrix, labels)
        
        print(loss)

        return loss
