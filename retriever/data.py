import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from open_clip import tokenize

class Flickr30kDataset(Dataset):
    def __init__(self, json_path, img_folder, split="train", transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = []
        self.img_folder = img_folder
        self.transform = transform

        for img_data in data['images']:
            if img_data.get("split", "train") != split:
                continue
            img_path = os.path.join(self.img_folder, img_data['filename'])
            for sentence in img_data['sentences']:
                text = sentence['raw']
                self.samples.append((img_path, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, text


def build_data(args, transform=None):
    json_path = args.data_annotation_path
    img_folder = args.data_image_path

    train_set = Flickr30kDataset(json_path, img_folder, split="train", transform=transform)
    val_set = Flickr30kDataset(json_path, img_folder, split="val", transform=transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader
    }
