from sentence_transformers import SentenceTransformerTrainer
from typing import Optional, Union, Dict, Any, List


class ContrastiveSentenceTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[bool, str]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs
    ):
        pass
