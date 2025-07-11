import faiss
import sys
import numpy as np
from tqdm import tqdm
import torch
import json
import os

from retriever.model import build_model
from retriever.data import build_data
from retriever.params import parse_args

@torch.no_grad()
def build_faiss_index(model, dataloader, device, faiss_dir="data/faiss_index", use_gpu=False):
    model.eval()
    image_feats = []
    image_paths = dataloader['val_image_paths']

    for batch in tqdm(dataloader['val_loader'], desc="Building FAISS index"):
        images, _ = batch
        images = images.to(device)

        with torch.autocast(device_type='cuda'):
            feats = model.encode_image(images)
        feats = feats.float().cpu().numpy()

        image_feats.append(feats)

    # Stack all image features
    image_feats = np.vstack(image_feats)

    # Normalize for cosine similarity
    faiss.normalize_L2(image_feats)

    dim = image_feats.shape[1]
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatIP(dim)  # Cosine similarity

    index.add(image_feats)

    os.makedirs(faiss_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(faiss_dir, "image.index"))

    with open(os.path.join(faiss_dir, "image_paths.txt"), "w") as f:
        for p in image_paths:
            f.write(f"{p}\n")

    print(f"FAISS index built with {len(image_feats)} vectors.")
    
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, preprocess = build_model(args.model_name)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best.pt")))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    dataloader = build_data(args, preprocess)
    build_faiss_index(model, dataloader, device, args.faiss_dir)