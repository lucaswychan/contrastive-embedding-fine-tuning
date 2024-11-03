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

    # override
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        dataset_name = inputs.pop("dataset_name", None)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]
            
            
        if isinstance(inputs, dict):
            queries = inputs.get("query", None)
            positives = inputs.get("positive", None)
            labels = inputs.get("label", None) if self.use_labels else None
        else:
            raise ValueError("Batch format not recognized")
        
        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and model != self.model  # Only if the model is wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model != model  # Only if the wrapped model is not already stored
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)

        # Generate embeddings
        query_embeddings = model.encode(
            queries, convert_to_tensor=True, batch_size=len(queries)
        )
        positive_embeddings = model.encode(
            positives, convert_to_tensor=True, batch_size=len(positives)
        )

        # Convert labels to tensor if using them
        if labels is not None:
            labels = torch.tensor(labels, device=self.device)

        # Compute loss
        loss = loss_fn(query_embeddings, positive_embeddings, labels)
        
        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}
        
        return loss
