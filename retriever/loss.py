import torch
import torch.nn.functional as F

def contrastive_loss(image_feat, text_feat):
    # Normalize
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)

    # Compute similarity scores
    logits_per_image = image_feat @ text_feat.T  # [N, N]
    logits_per_text = text_feat @ image_feat.T  # [N, N]

    # Ground-truth: diagonal is the correct pair
    labels = torch.arange(image_feat.size(0), device=image_feat.device)

    # Cross entropy in both directions
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    return (loss_i2t + loss_t2i) / 2
