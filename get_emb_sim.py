from sentence_transformers import SentenceTransformer
import json
import torch
from utils import get_available_gpu_idx
import logging

available_gpu_idx = get_available_gpu_idx()
if available_gpu_idx is None:
    raise ValueError("No available GPU found!")

available_cuda = f"cuda:{available_gpu_idx}"
print(f"Using GPU: {available_cuda}")

def get_emb_sim(source, target):
    device = torch.device(available_cuda)
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    
    source_emb = model.encode(source, convert_to_tensor=True)
    target_emb = model.encode(target, convert_to_tensor=True)
    
    sim = torch.cosine_similarity(source_emb, target_emb, dim=-1).item()
    
    return sim


if __name__ == "__main__":
    # with open("data/covered_entity.json", "r") as f:
    #     data = json.load(f)
    
    # source = "Covered entity"

    # f = open("data/covered_entity.txt", "w")
    
    # for context in data["context"]:
    #     target = context
    #     sim = get_emb_sim(source, target)
        
    #     f.write(f"{source}\t{target}  ---  {sim}\n")
    
    # f.close()
    
    # sim = get_emb_sim("HIPAA", "medical record")
    """
    # HIPAA vs medical record = 0.47049009799957275
    # HIPAA vs privacy regulation = 0.47049009799957275
    """
    
    sim = get_emb_sim("HIPAA", "privacy regulation")
    
    print(sim)
        
        
    
    
    
    
