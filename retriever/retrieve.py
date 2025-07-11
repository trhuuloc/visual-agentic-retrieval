import faiss
import sys
import torch
import os

from retriever.model import build_model
from retriever.params import parse_args
from open_clip import tokenize

@torch.no_grad()
def search_faiss(model, query_texts, index_path="./data/faiss_index/image.index", image_paths='./data/faiss_index/image_paths.txt', top_k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = faiss.read_index(index_path)

    text_inputs = tokenize(query_texts).to(device)
    with torch.autocast(device_type='cuda'):
        text_feats = model.encode_text(text_inputs)
    text_feats = text_feats.float().cpu().numpy()
    faiss.normalize_L2(text_feats)

    # Search
    D, I = index.search(text_feats, top_k)

    # Load image paths
    with open(image_paths, 'r') as f:
        all_image_paths = [line.strip() for line in f.readlines()]

    # Map FAISS indices to actual image paths
    retrieved_paths = []
    for indices in I:
        paths = [all_image_paths[i] for i in indices]
        retrieved_paths.append(paths)

    return D, I, retrieved_paths

    
if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, preprocess = build_model(args.model_name)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, "best.pt")))
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    index_path = args.faiss_dir + '/image.index'
    image_paths = args.faiss_dir + '/image_paths.txt'
    
    text = 'A male'
    _, _, a = search_faiss(model, text, index_path, image_paths, args.top_k)
    print(a)