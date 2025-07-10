import argparse


def parse_args(args):
    parser = argparse.ArgumentParser()
    
    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default='ViT-B-16',
        help='CLIP pretrained version'
    )
    
    # Hyperparameter Training
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'

    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='Weight decay'
    )
    
    # Dataset
    parser.add_argument(
        '--data_image_path',
        type=str,
        default="./data/flickr30k/Images",
        help='Dataset image path'
    )
    parser.add_argument(
        '--data_annotation_path',
        type=str,
        default="./data/flickr30k/dataset_flickr30k.json",
        help='Dataset annotation'
    )
    
    # Other
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='ckpts/',
        help='Save checkpoints'
    )
    args = parser.parse_args(args)
    
    return args