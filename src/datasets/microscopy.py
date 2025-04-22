import os
import tifffile
import torch
from torch.utils.data import Dataset
import json
import numpy as np

class MicroscopyDataset(Dataset):
    def __init__(self, root_dir, transform=None, stats_file='dataset_stats.json'):
        """
        Microscopy dataset loader for 5-channel TIFF images.
        
        Args:
            root_dir (str): Path to the root directory containing class folders
            transform (callable, optional): Optional transform to be applied on a sample
            stats_file (str): Path to the JSON file containing dataset statistics
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load dataset statistics
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        self.quantiles = torch.tensor(stats['quantiles'])
        self.means = torch.tensor(stats['means'])
        self.stds = torch.tensor(stats['stds'])
        
        import code; code.interact(local=dict(globals(), **locals()))
        # Get class information
        self.classes = [file for file in sorted(os.listdir(root_dir)) if file.endswith('.tiff')]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all samples
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.tiff'):
                    self.samples.append((
                        os.path.join(class_dir, img_name),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        
        # Load 5-channel TIFF image
        image = tifffile.imread(img_path)  # Shape: (50, 50, 5)
        
        # Convert to torch tensor and normalize
        image = torch.from_numpy(image).float()
        
        # Clip to 99% quantile for each channel
        for channel in range(5):
            image[..., channel] = torch.clamp(image[..., channel], 0, self.quantiles[channel])
        
        # Normalize
        image = (image - self.means) / self.stds
        
        # Permute to (C, H, W) format
        image = image.permute(2, 0, 1)
        
        if self.transform:
            image = self.transform(image)
            
        return image, target

def make_microscopy_dataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True
):
    """
    Create microscopy dataset and dataloader.
    
    Args:
        transform: Transform to apply to images
        batch_size: Batch size
        collator: Optional collator function
        pin_mem: Whether to pin memory
        num_workers: Number of workers for data loading
        world_size: Number of processes in distributed training
        rank: Process rank in distributed training
        root_path: Path to dataset root
        training: Whether to use training set
        drop_last: Whether to drop last incomplete batch
    """
    dataset = MicroscopyDataset(
        root_dir=root_path,
        transform=transform
    )
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    
    return dataset, data_loader, dist_sampler 