# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
sys.path.insert(0, '../src/')
import json
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from utils import seed_hash
from algorithms import get_algorithm_class, AlgorithmModule
from hparams_registry import default_hparams, random_hparams
from data import DataModule


def get_parser():
    parser = argparse.ArgumentParser(description='Domain Generalization Training')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--algorithm', type=str, required=True)
    parser.add_argument('--model_arch', type=str, default="resnet50")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])

    # Hyperparameter arguments
    parser.add_argument('--hparams', type=str, default=None,
                      help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                      help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                      help='Trial number (used for seeding split_dataset and random_hparams)')

    # Training arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--checkpoint_freq', type=int, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0.)

    # Transfer learning arguments
    parser.add_argument('--transfer', action='store_true',
                      help='Whether to use transfer learning (only train last layer)')
    parser.add_argument('--weights', type=str, default=None,
                      help='Weight initialization strategy')

    # Logging arguments
    parser.add_argument('--log_backend', type=str, default='none',
                      choices=['tensorboard', 'wandb', 'csv', 'none'])

    return parser

def train(args):
    """Main training function."""
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # added this which seems to make things faster; DM Haoran if you get an issue with "too many open files"
    torch.multiprocessing.set_sharing_strategy('file_descriptor')

    # Setup hyperparameters
    if args.hparams_seed == 0:
        hparams = default_hparams(args.algorithm, args.dataset)
    else:
        hparams = random_hparams(
            args.algorithm,
            args.dataset,
            seed_hash(args.hparams_seed, args.trial_seed, args.model_arch, args.transfer, args.weights)
        )

    if args.hparams:
        hparams.update(json.loads(args.hparams))

    # Add transfer learning and weights settings to hparams
    hparams.update({
        'args': args.__dict__,
        'transfer': args.transfer,
        'weights': args.weights,
        'model_arch': args.model_arch,
    })

    # Set random seeds
    pl.seed_everything(args.seed, workers=True)

    # Setup logging
    logger = None
    if args.log_backend == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=args.output_dir,
            default_hp_metric=False
        )
    elif args.log_backend == "wandb":
        logger = WandbLogger(
            project="oodselect_training",
            name=f"{args.algorithm}_{args.dataset}_{args.model_arch}_{args.weights}{'_transfer' if args.transfer else ''}_{args.seed}_{args.hparams_seed}_{args.trial_seed}",
            config=args.__dict__,
            save_dir=args.output_dir
        )
    elif args.log_backend == "csv":
        logger = CSVLogger(
            save_dir=args.output_dir,
            name="lightning_logs"
        )
    elif args.log_backend == "none":
        # Logger is just print to console
        logger = None
    else:
        raise ValueError(f"Invalid log backend: {args.log_backend}")

    # Create data module first to access dataset properties
    dm = DataModule(args, hparams)
    dm.setup()

    # Setup callbacks
    callbacks = [
    ]

    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        accelerator='gpu',
        devices=1,
        max_steps=args.steps or dm.dataset.N_STEPS,
        val_check_interval=args.checkpoint_freq or dm.dataset.CHECKPOINT_FREQ,  # Use dataset's checkpoint frequency
        enable_checkpointing=False,
        log_every_n_steps=args.checkpoint_freq or dm.dataset.CHECKPOINT_FREQ,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks,
    )

    # Create model
    model = AlgorithmModule(
        algorithm_class=get_algorithm_class(args.algorithm),
        input_shape=dm.dataset.input_shape,  # type: ignore
        num_classes=dm.dataset.num_classes,  # type: ignore
        num_domains=len(dm.dataset) - len(args.test_envs),  # type: ignore
        hparams=hparams
    )

    # Train
    trainer.fit(model, dm)

    # Mark job as done
    with open(os.path.join(args.output_dir, "done"), "w") as f:
        f.write("")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))
    train(args)

