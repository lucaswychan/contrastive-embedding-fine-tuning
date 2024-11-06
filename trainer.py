import json
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer


class ContrastiveSTTrainer(SentenceTransformerTrainer):
    def __init__(self, use_labels: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_labels = use_labels
        print(f"In Trainer: Use labels - {self.use_labels}")

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
            queries = inputs.get("query_input_ids", None)
            queries_attention_mask = inputs.get("query_attention_mask", None)

            query_inputs = {
                "input_ids": queries,
                "attention_mask": queries_attention_mask,
            }

            # print(f"Size of queries: {queries.size()}")
            # print(f"Size of queries attention mask: {queries_attention_mask.size()}")

            positives = inputs.get("positive_input_ids", None)
            positives_attention_mask = inputs.get("positive_attention_mask", None)

            # print(f"Size of positives: {positives.size()}")
            # print(f"Size of positives attention mask: {positives_attention_mask.size()}")

            positive_inputs = {
                "input_ids": positives,
                "attention_mask": positives_attention_mask,
            }

            labels = inputs.get("label", None) if self.use_labels else None
            print(f"Size of labels: {labels.size() if labels is not None else None}")
        else:
            raise ValueError("Batch format not recognized")

        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and model != self.model  # Only if the model is wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model
            != model  # Only if the wrapped model is not already stored
        ):
            print(
                "Storing wrapped model in loss function. "
                "This is only done once per training process."
            )
            loss_fn = self.override_model_in_loss(loss_fn, model)

        # Convert labels to tensor if using them
        if labels is not None:
            labels = labels.clone().detach()

        # Compute loss
        loss = loss_fn([query_inputs, positive_inputs], labels)

        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}

        return loss
