"""
Visualization utilities for I-JEPA.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple
import time
import logging
import wandb
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

# Global storage for sampled indices to ensure consistent visualization
_UMAP_SAMPLED_INDICES = None

def create_balanced_sample_indices(dataset, max_samples=1000):
    """
    Create a balanced sample of indices from each class.
    
    Args:
        dataset: Dataset with class information
        max_samples: Maximum total samples to return
        
    Returns:
        indices: List of indices to sample from the dataset
    """
    global _UMAP_SAMPLED_INDICES
    
    # If we already have sampled indices, return them
    if _UMAP_SAMPLED_INDICES is not None:
        logger.info(f"Using existing sampled indices for visualization")
        return _UMAP_SAMPLED_INDICES
    
    logger.info(f"Creating balanced sample of indices for visualization")
    
    # Check if dataset has classes attribute
    if not hasattr(dataset, 'class_to_idx'):
        logger.warning("Dataset doesn't have 'class_to_idx' attribute, using random sampling")
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        _UMAP_SAMPLED_INDICES = indices[:max_samples]
        return _UMAP_SAMPLED_INDICES
    
    # Group indices by class
    class_indices = defaultdict(list)
    
    # For microscopy dataset, we can directly access the samples
    if hasattr(dataset, 'samples'):
        for idx, (_, class_idx) in enumerate(dataset.samples):
            class_indices[class_idx].append(idx)
    else:
        # For other datasets, we need to iterate through the dataset
        for idx in range(len(dataset)):
            _, class_idx = dataset[idx]
            class_indices[class_idx].append(idx)
    
    # Determine how many samples per class
    num_classes = len(class_indices)
    samples_per_class = max(1, max_samples // num_classes)
    
    # Create a balanced sample
    balanced_indices = []
    for class_idx, indices in class_indices.items():
        # Randomly sample from this class
        sampled = random.sample(indices, min(samples_per_class, len(indices)))
        balanced_indices.extend(sampled)
    
    # Shuffle the combined indices
    random.shuffle(balanced_indices)
    
    # Limit to max_samples
    balanced_indices = balanced_indices[:max_samples]
    
    logger.info(f"Created balanced sample with {len(balanced_indices)} indices from {num_classes} classes")
    
    # Store for future use
    _UMAP_SAMPLED_INDICES = balanced_indices
    
    return balanced_indices

def extract_embeddings(
    encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: Optional[int] = None,
    normalize: bool = True,
    use_balanced_sampling: bool = True,
    dataset = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from a dataset using the encoder.
    
    Args:
        encoder: The encoder model to use
        dataloader: DataLoader containing images
        device: Device to run the encoder on
        max_samples: Maximum number of samples to process
        normalize: Whether to normalize embeddings with layer norm
        use_balanced_sampling: Whether to use balanced sampling across classes
        dataset: The dataset, needed for balanced sampling
        
    Returns:
        embeddings: Array of embeddings
        labels: Array of labels
    """
    encoder.eval()
    
    # Use balanced sampling if requested and dataset is provided
    if use_balanced_sampling and dataset is not None:
        return extract_embeddings_balanced(
            encoder=encoder,
            dataset=dataset,
            device=device,
            max_samples=max_samples,
            normalize=normalize
        )
    
    # Otherwise, use the standard dataloader approach
    embeddings = []
    labels = []
    sample_count = 0
    
    logger.info(f"Extracting embeddings from dataset...")
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            # Move to device
            images = images.to(device, non_blocking=True)
            
            # Get embeddings - don't use any masking
            batch_embeddings = encoder(images)
            
            # Apply layer norm if needed
            if normalize:
                batch_embeddings = torch.nn.functional.layer_norm(
                    batch_embeddings, 
                    (batch_embeddings.size(-1),)
                )
            
            # Convert to numpy and store
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.append(batch_embeddings)
            labels.append(batch_labels.cpu().numpy())
            
            # Update count and check if we've reached max_samples
            sample_count += images.shape[0]
            if max_samples and sample_count >= max_samples:
                break
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    
    # If we have more than requested, truncate
    if max_samples and embeddings.shape[0] > max_samples:
        embeddings = embeddings[:max_samples]
        labels = labels[:max_samples]
    
    logger.info(f"Extracted embeddings with shape {embeddings.shape}")
    return embeddings, labels

def extract_embeddings_balanced(
    encoder: torch.nn.Module,
    dataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from a dataset using balanced sampling across classes.
    
    Args:
        encoder: The encoder model to use
        dataset: Dataset to sample from
        device: Device to run the encoder on
        max_samples: Maximum number of samples to process
        normalize: Whether to normalize embeddings with layer norm
        
    Returns:
        embeddings: Array of embeddings
        labels: Array of labels
    """
    # Get balanced sample indices
    indices = create_balanced_sample_indices(dataset, max_samples)
    
    embeddings = []
    labels = []
    
    logger.info(f"Extracting embeddings from {len(indices)} balanced samples...")
    
    with torch.no_grad():
        for idx in indices:
            # Get sample directly from dataset
            image, label = dataset[idx]
            
            # Add batch dimension and move to device
            image = image.unsqueeze(0).to(device, non_blocking=True)
            
            # Get embeddings - don't use any masking
            embedding = encoder(image).mean(dim=1) # average pooling over tokens
            
            # Apply layer norm if needed
            if normalize:
                embedding = torch.nn.functional.layer_norm(
                    embedding, 
                    (embedding.size(-1),)
                )
            
            # Convert to numpy and store
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
            labels.append(label)
    
    # Concatenate all samples
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    logger.info(f"Extracted balanced embeddings with shape {embeddings.shape}")
    return embeddings, labels

def create_umap_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray, 
    label_names: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    title: str = "UMAP of Image Embeddings",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a UMAP visualization of embeddings.
    
    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        labels: Class labels for coloring points
        label_names: Optional class names corresponding to labels
        n_neighbors: UMAP hyperparameter for local neighborhood size
        min_dist: UMAP hyperparameter for minimum distance between points
        title: Plot title
        save_path: Path to save the visualization
        
    Returns:
        fig: The generated matplotlib figure
    """
    # Import UMAP here to avoid issues if it's not installed but other functions are used
    try:
        from umap import UMAP
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.error("Please install umap-learn and scikit-learn: pip install umap-learn scikit-learn")
        raise
    
    start_time = time.time()
    logger.info(f"Starting UMAP projection with {embeddings.shape[0]} samples...")
    
    # Standardize embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Initialize and fit UMAP
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    
    umap_embeddings = umap_model.fit_transform(scaled_embeddings)
    
    logger.info(f"UMAP projection completed in {time.time() - start_time:.2f} seconds")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set up colors for scatter plot
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Use categorical colormap for discrete classes
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # Create scatter plot
    scatter = ax.scatter(
        umap_embeddings[:, 0],
        umap_embeddings[:, 1],
        c=labels,
        cmap=cmap,
        alpha=0.7,
        s=10
    )
    
    # Add colorbar with class names if provided
    if label_names is not None:
        # Create a legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=cmap(i), markersize=8, label=label_names[i]) 
                          for i in range(len(label_names))]
        ax.legend(handles=legend_elements, loc='best', title="Classes")
    else:
        # Just add a colorbar
        plt.colorbar(scatter, ax=ax, label="Class")
    
    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.tight_layout()
    
    return fig

def log_umap_to_wandb(
    encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    epoch: int,
    max_samples: int = 1000,
    save_dir: Optional[str] = None
) -> None:
    """
    Extract embeddings, create UMAP visualization, and log to W&B.
    
    Args:
        encoder: Model to extract embeddings with
        dataloader: DataLoader for the dataset
        dataset: Dataset containing class information
        device: Device to run inference on
        epoch: Current epoch number
        max_samples: Maximum number of samples to include
        save_dir: Directory to save plots locally (optional)
    """
    if not wandb.run:
        logger.warning("W&B not initialized, skipping UMAP visualization")
        return
    
    # Check if dataset has classes attribute
    if not hasattr(dataset, 'classes'):
        logger.warning("Dataset doesn't have 'classes' attribute, using numeric labels")
        label_names = None
    else:
        label_names = dataset.classes
    
    # Extract embeddings using balanced sampling
    embeddings, labels = extract_embeddings(
        encoder=encoder,
        dataloader=dataloader,
        dataset=dataset,
        device=device,
        max_samples=max_samples,
        normalize=True,
        use_balanced_sampling=True
    )
    
    # Create plot
    title = f"UMAP of Embeddings (Epoch {epoch})"
    
    # Determine save path if needed
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"umap_epoch_{epoch}.png")
    
    # Generate UMAP visualization
    fig = create_umap_visualization(
        embeddings=embeddings,
        labels=labels,
        label_names=label_names,
        title=title,
        save_path=save_path
    )
    
    # Log to W&B
    wandb.log({
        "umap_plot": wandb.Image(fig),
        "epoch": epoch
    })
    
    # Close the figure to avoid memory leaks
    plt.close(fig)