from sentence_embedding.emb_model import BaseSentenceEmbeddingModel, HFModel, SonarModel


class SentenceEmbeddingModelFactory:
    @staticmethod
    def get_model(
        model_name_or_path: str, device=None, **kwargs
    ) -> BaseSentenceEmbeddingModel:
        if model_name_or_path == "sonar":
            return SonarModel(device=device)
        elif model_name_or_path == "hf":
            return HFModel(kwargs.get("model", "all-MiniLM-L6-v2"), device=device)
        else:
            raise ValueError(f"Unknown model: {model_name_or_path}")
