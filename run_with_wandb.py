#!/usr/bin/env python3
"""
Script to run I-JEPA training with Weights and Biases integration.
"""

import argparse
import os
import sys
import logging

from main import process_main, single_process_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description='Run I-JEPA training with Weights and Biases')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file')
    parser.add_argument(
        '--devices', type=str, nargs='+', default=['cuda:0'],
        help='Which devices to use on local machine')
    parser.add_argument(
        '--wandb_project', type=str, default='ijepa',
        help='Weights & Biases project name')
    parser.add_argument(
        '--wandb_entity', type=str, default=None,
        help='Weights & Biases entity (username or team name)')
    parser.add_argument(
        '--wandb_run_name', type=str, default=None,
        help='Name for this run in Weights & Biases')
    parser.add_argument(
        '--wandb_tags', type=str, nargs='+', default=None,
        help='Tags for this run in Weights & Biases')
    args = parser.parse_args()

    # Ensure the config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Update config file in memory to include W&B settings
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with W&B info
    if 'meta' not in config:
        config['meta'] = {}
    
    config['meta']['use_wandb'] = True
    if args.wandb_project:
        config['meta']['wandb_project'] = args.wandb_project
    if args.wandb_entity:
        config['meta']['wandb_entity'] = args.wandb_entity
    if args.wandb_run_name:
        config['meta']['wandb_run_name'] = args.wandb_run_name
    
    # Write updated config to a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp:
        yaml.dump(config, temp)
        temp_config_path = temp.name
    
    try:
        # Run the training
        num_gpus = len(args.devices)
        logger.info(f"Starting I-JEPA training with {num_gpus} GPUs")
        logger.info(f"Weights & Biases project: {config['meta'].get('wandb_project', 'ijepa')}")
        
        # Run in single or multi-GPU mode
        if num_gpus == 1:
            single_process_main(temp_config_path, args.devices[0])
        else:
            import multiprocessing as mp
            mp.set_start_method('spawn')
            for rank in range(num_gpus):
                mp.Process(
                    target=process_main,
                    args=(rank, temp_config_path, num_gpus, args.devices)
                ).start()
    finally:
        # Clean up the temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == '__main__':
    main() 