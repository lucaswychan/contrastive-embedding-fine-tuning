import os
import sys
from abc import ABC, abstractmethod

sys.path.append(
    os.path.abspath(os.path.join(os.path.pardir, "contrastive-embedding-fine-tuning"))
)

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from utils import get_available_gpu

# from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


class BaseSentenceEmbeddingModel(ABC):
    def __init__(self, device):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device(get_available_gpu(use_cpu=True))

    @abstractmethod
    def encode(self, sentences: str) -> torch.Tensor:
        raise NotImplementedError


class HFModel(BaseSentenceEmbeddingModel):
    def __init__(self, model_name_or_path: str, device=None):
        super().__init__(device)
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

    def encode(self, sentences) -> torch.Tensor:
        return self.model.encode(
            sentences,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False,
        )


class SonarModel(BaseSentenceEmbeddingModel):
    def __init__(self, device=None):
        super().__init__(device)
        # self.model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")
        self.model = None

    def encode(self, sentences) -> torch.Tensor:
        return self.model.predict(sentences, source_lang="eng_Latn")
