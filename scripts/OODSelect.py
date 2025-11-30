#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.insert(0, '../src/')
import argparse
os.environ["WANDB_MODE"] = "offline"
import wandb

import torch

import OODSelect_utils
from OODSelect_utils import load_prediction_data, prepare_dataset, run_optuna, extract_data
from datasets import get_dataset_class


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning and testing')
    parser.add_argument('--dataset', type=str, default='TerraIncognita',
                      help='Dataset name (default: TerraIncognita)')
    parser.add_argument('--results_dir', type=str,
                      default="",
                      help='Directory containing results')
    parser.add_argument('--loss_type', type=str, default='r',
                      choices=['r', 'r2'],
                      help='Type of loss function (default: r)')
    parser.add_argument('--num_epochs', type=int, default=3000,
                      help='Number of epochs for training (default: 3000)')
    parser.add_argument('--max_total_samples', type=int,
                      help='Maximum number of samples to use (default: None)')
    parser.add_argument('--max_total_models', type=int,
                      help='Maximum number of models to use (default: None)')
    parser.add_argument('--num_trials', type=int, default=20,
                      help='Number of trials for hyperparameter optimization (default: 20)')
    parser.add_argument('--train_idxs', type=int, nargs="+", default=[0, 1, 2],
                      help='Indices of training sets to use (default: 0,1,2)')
    parser.add_argument('--test_idx', type=int, default=3,
                      help='Index of test set to use (default: 3)')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory to save outputs (default: ./results)')
    parser.add_argument('--num_OOD_samples', type=int, default=None,
                      help='Number of OOD samples to use (default: None)')
    parser.add_argument('--wandb_project', type=str, default='oodselect',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                      help='Weights & Biases entity (username)')
    parser.add_argument('--seed', type=int, default=0,
                      help='Random seed (default: 0)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    args.train_idxs = '-'.join([str(i) for i in sorted(args.train_idxs)])
    args.test_idx = str(args.test_idx)

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # Configurations
    dataset = args.dataset
    num_domains = len(get_dataset_class(args.dataset).ENVIRONMENTS)
    results_dir = args.results_dir

    # Load predictions and configuration
    # X is matrix of shape num_models x total_num_OOD_samples where the columns binary flag on if that sample is correctly predicted
    predictions_data, acc_cols = load_prediction_data(
        dataset=dataset,
        num_domains=num_domains,
        results_dir=results_dir,
        train_idxs=args.train_idxs,
        test_idx=args.test_idx,
        max_total_samples=args.max_total_samples,
        max_total_models=args.max_total_models,
        seed=args.seed
    )

    X, y = extract_data(predictions_data, acc_cols)

    assert X.shape[0] > 20, f"X.shape[0] = {X.shape[0]}"

    if args.num_OOD_samples > X.shape[1]:
        print(f"args.num_OOD_samples = {args.num_OOD_samples} > X.shape[1] = {X.shape[1]}")
        exit()

    if args.num_OOD_samples < 0:
        args.num_OOD_samples = X.shape[1]

    # Prepare training, validation, and test datasets
    train_loader, val_loader, _, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(X, y)

    # num_samples is the number of columns in our OOD_correct_flags (which equals min_n)
    num_samples = X_train_tensor.shape[1]

    if num_samples < args.num_OOD_samples:
        args.num_OOD_samples = num_samples

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project=f"{args.dataset}_{args.train_idxs}_{args.test_idx}_{args.loss_type}_{args.wandb_project}",
        entity=args.wandb_entity,
        config=args.__dict__,
    )

    # Run hyperparameter tuning via Optuna.
    study = run_optuna(
        train_loader,
        val_loader,
        args.num_epochs,
        num_samples,
        args.num_OOD_samples,
        args.loss_type,
        n_trials=args.num_trials,
        output_dir=args.output_dir
    )

    # Log best trial details to wandb
    wandb.log({
        "best_trial_value": study.best_trial.value,
        "best_trial_params": study.best_trial.params,
        "best_trial_number": study.best_trial.number
    })

    # Display best trial details
    print(study.best_trial)

    # Test the best model on the test set.
    test_corr = OODSelect_utils.TestSetFinder.test(
        X_test_tensor,
        y_test_tensor,
        torch.tensor(study.best_trial.user_attrs["soft_selection"]),
        args.num_OOD_samples,
        args.loss_type
    )
    val_corr = OODSelect_utils.TestSetFinder.test(
        X_val_tensor,
        y_val_tensor,
        torch.tensor(study.best_trial.user_attrs["soft_selection"]),
        args.num_OOD_samples,
        args.loss_type
    )
    train_corr = OODSelect_utils.TestSetFinder.test(
        X_train_tensor,
        y_train_tensor,
        torch.tensor(study.best_trial.user_attrs["soft_selection"]),
        args.num_OOD_samples,
        args.loss_type
    )

    print(f"Test correlation: {test_corr}")

    # Log test correlation to wandb
    wandb.log({"hard_test_correlation": test_corr})
    wandb.log({"hard_val_correlation": val_corr})
    wandb.log({"hard_train_correlation": train_corr})

    # Save model state
    model_state_path = os.path.join(
        args.output_dir, f"min_acc_selection_{args.loss_type}.pt")
    torch.save(study.best_trial.user_attrs["soft_selection"],
               model_state_path)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    # Log model state to wandb
    os.makedirs(os.path.join(args.output_dir, 'wandb'), exist_ok=True)
    wandb.save(os.path.join(args.output_dir, 'wandb', '*.pt'))

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

