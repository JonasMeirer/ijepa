import os
import numpy as np
import tifffile
from tqdm import tqdm
import torch
from pathlib import Path

def calculate_dataset_stats(root_dir):
    """
    Calculate normalization statistics for a microscopy dataset.
    
    Args:
        root_dir (str): Path to the root directory containing class folders
        
    Returns:
        tuple: (quantiles, means, stds) for each channel
    """
    # Initialize accumulators
    all_pixels = [[] for _ in range(5)]  # For 5 channels
    img_means = [[] for _ in range(5)]
    img_vars = [[] for _ in range(5)]

    # Get all TIFF files
    tiff_files = []
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            tiff_files.extend([
                os.path.join(class_dir, f) for f in os.listdir(class_dir)
                if f.endswith('.tiff')
            ])
    
    print(f"Found {len(tiff_files)} images")
    
    # First pass: collect all pixel values for quantile calculation
    for img_path in tqdm(tiff_files, desc="Collecting pixel values"):
        img = tifffile.imread(img_path)  # Shape: (50, 50, 5)
        for channel in range(5):
            all_pixels[channel].extend(img[..., channel].flatten())
    
    # Calculate 99% quantiles for each channel
    quantiles = [np.round(np.quantile(channel_pixels, 0.99), 4).item() for channel_pixels in all_pixels]
    print(f"99% quantiles: {quantiles}")
    
    # Second pass: calculate mean and std after clipping
    for img_path in tqdm(tiff_files, desc="Calculating statistics"):
        img = tifffile.imread(img_path)
        # Clip each channel to its 99% quantile
        for channel in range(5):
            img[..., channel] = np.clip(img[..., channel], 0, quantiles[channel])
        
            # Update accumulators
            img_means[channel].append(img[..., channel].mean(axis=(0, 1)))
            img_vars[channel].append(img[..., channel].var(axis=(0, 1)))
    
    # Calculate final statistics
    means = np.array([np.mean(channel_means) for channel_means in img_means]).round(4)
    stds = np.sqrt(np.array([np.mean(channel_vars) for channel_vars in img_vars])).round(4)
    
    return quantiles, means.tolist(), stds.tolist()

if __name__ == "__main__":
    root_dir = "data/samples"  # Update this path if needed
    quantiles, means, stds = calculate_dataset_stats(root_dir)
    
    print("\nDataset Statistics:")
    print(f"99% Quantiles: {quantiles}")
    print(f"Means: {means}")
    print(f"Standard Deviations: {stds}")
    
    # Save statistics to a file
    stats = {
        'quantiles': quantiles,
        'means': means,
        'stds': stds
    }
    
    import json
    with open('dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4) 