import json
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer


class ContrastiveSTTrainer(SentenceTransformer):
    def __init__(self, use_labels: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_labels = use_labels

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if isinstance(inputs, dict):
            queries = inputs.get("query", None)
            positives = inputs.get("positive", None)
            labels = inputs.get("label", None) if self.use_labels else None
        else:
            raise ValueError("Batch format not recognized")

        # Generate embeddings
        query_embeddings = self.model.encode(
            queries, convert_to_tensor=True, batch_size=len(queries)
        )
        positive_embeddings = self.model.encode(
            positives, convert_to_tensor=True, batch_size=len(positives)
        )

        # Convert labels to tensor if using them
        if labels is not None:
            labels = torch.tensor(labels, device=self.device)

        # Compute loss
        loss = self.loss(query_embeddings, positive_embeddings, labels)
        return loss
