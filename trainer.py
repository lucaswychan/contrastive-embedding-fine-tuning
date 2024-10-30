from sentence_transformers import SentenceTransformerTrainer

class ContrastiveSentenceTrainer(SentenceTransformerTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self):
        pass
    