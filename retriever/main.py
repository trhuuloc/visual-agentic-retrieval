import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from open_clip import tokenize

from retriever.model import build_model
from retriever.params import parse_args
from retriever.data import build_data
from retriever.loss import contrastive_loss

def train(args):
    args = parse_args(args)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # Build model, data
    model, preprocess = build_model(args.model_name)
    model = model.to(device)

    dataloader = build_data(args, preprocess)
    train_loader, val_loader = dataloader['train_loader'], dataloader['val_loader']

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    # Create checkpoint folder
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")

    for i, group in enumerate(optimizer.param_groups):
        for p in group['params']:
            assert p.device.type == 'cuda', f"Param {i} not on GPU"
        
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_loader):
            image, text = batch
            text = tokenize(text).to(device)
            image = image.to(device)

            with torch.autocast(device_type='cuda'):
                image_feat = model.encode_image(image)
                text_feat = model.encode_text(text)
                loss = contrastive_loss(image_feat, text_feat)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        val_loss = eval(model, val_loader, device)

        print(f"\nEpoch {epoch+1} Completed:")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")

        scheduler.step()
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(args.ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Saved best checkpoint to {best_ckpt_path}")

        # Save last checkpoint
        last_ckpt_path = os.path.join(args.ckpt_dir, "last.pt")
        torch.save(model.state_dict(), last_ckpt_path)
        print(f"Saved last checkpoint to {last_ckpt_path}")


@torch.no_grad()
def eval(model, val_loader, device):
    model.eval()
    total_loss = 0

    for batch in val_loader:
        image, text = batch
        text = tokenize(text).to(device)
        image = image.to(device)

        with torch.autocast(device_type='cuda'):
            image_feat = model.encode_image(image)
            text_feat = model.encode_text(text)
            loss = contrastive_loss(image_feat, text_feat)
            
        total_loss += loss.item()
    return total_loss / len(val_loader)


if __name__ == "__main__":
    train(sys.argv[1:])
