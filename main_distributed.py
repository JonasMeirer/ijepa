# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import logging
import os
import pprint
import sys
import yaml

import submitit

from src.train import main as app_main

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', type=str,
    help='location to save submitit logs')
parser.add_argument(
    '--batch-launch', action='store_true',
    help='whether fname points to a file to batch-lauch several config files')
parser.add_argument(
    '--fname', type=str,
    help='yaml file containing config file names to launch',
    default='configs.yaml')
parser.add_argument(
    '--partition', type=str,
    help='cluster partition to submit jobs on')
parser.add_argument(
    '--nodes', type=int, default=1,
    help='num. nodes to request for job')
parser.add_argument(
    '--tasks-per-node', type=int, default=1,
    help='num. procs to per node')
parser.add_argument(
    '--time', type=int, default=4300,
    help='time in minutes to run job')
parser.add_argument(
    '--use-wandb', action='store_true',
    help='whether to use Weights & Biases for logging')
parser.add_argument(
    '--wandb-project', type=str, default='ijepa',
    help='Weights & Biases project name')
parser.add_argument(
    '--wandb-entity', type=str, default=None,
    help='Weights & Biases entity (username or team name)')
parser.add_argument(
    '--wandb-run-name', type=str, default=None,
    help='Name for this run in Weights & Biases')


class Trainer:

    def __init__(self, config_fname):
        self.config_fname = config_fname

    def __call__(self):
        # -- load config file
        config = None
        with open(self.config_fname, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # -- update with Weights & Biases settings if specified
        if args.use_wandb:
            if 'meta' not in config:
                config['meta'] = {}
            config['meta']['use_wandb'] = True
            config['meta']['wandb_project'] = args.wandb_project
            config['meta']['wandb_entity'] = args.wandb_entity
            config['meta']['wandb_run_name'] = args.wandb_run_name
        # -- run real training
        app_main(args=config)
        return None


def launch():
    executor = submitit.AutoExecutor(
        folder=os.path.join(args.folder, 'job_%j'),
        slurm_max_num_timeout=20)
    executor.update_parameters(
        slurm_partition=args.partition,
        slurm_mem_per_gpu='55G',
        timeout_min=args.time,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        cpus_per_task=10,
        gpus_per_node=args.tasks_per_node)

    config_fnames = [args.fname]

    jobs, trainers = [], []
    with executor.batch():
        for cf in config_fnames:
            fb_trainer = Trainer(cf)
            job = executor.submit(fb_trainer,)
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


if __name__ == '__main__':
    args = parser.parse_args()
    launch()
