"""
Weights & Biases logger utilities for I-JEPA.
"""

import os
import wandb
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def init_wandb(project_name: str, 
               config: Dict[str, Any], 
               run_name: Optional[str] = None, 
               entity: Optional[str] = None, 
               group: Optional[str] = None,
               resume: bool = False) -> None:
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name: Name of the W&B project
        config: Configuration dictionary to log
        run_name: Optional name for this run
        entity: Optional W&B entity name (username or team name)
        group: Optional group name for this run (useful for comparing runs)
        resume: Whether to resume a previous run
    """
    if not wandb.run:
        # Generate run name if not provided
        if run_name is None:
            model_name = config['meta']['model_name']
            patch_size = config['mask']['patch_size']
            crop_size = config['data']['crop_size']
            batch_size = config['data']['batch_size']
            run_name = f"{model_name}.{patch_size}-{crop_size}px-bs{batch_size}"
        
        # Initialize W&B
        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=run_name,
            group=group,
            resume=resume,
            settings=wandb.Settings(start_method="thread")
        )
        logger.info(f"Initialized W&B run: {wandb.run.name}")

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number
    """
    if wandb.run:
        wandb.log(metrics, step=step)

def log_model(model_path: str, aliases: Optional[list] = None) -> None:
    """
    Log model checkpoint to W&B.
    
    Args:
        model_path: Path to the model checkpoint
        aliases: Optional list of aliases for this model
    """
    if wandb.run:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}", 
            type="model",
            description="I-JEPA model checkpoint"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=aliases)

def finish_run() -> None:
    """Finish W&B run."""
    if wandb.run:
        wandb.finish() 