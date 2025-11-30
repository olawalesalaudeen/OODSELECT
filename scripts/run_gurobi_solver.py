#!/usr/bin/env python
"""Run Gurobi solver to find optimal OOD sample selection."""

import argparse
import sys
sys.path.append('../src')
import os
from scipy.stats import pearsonr
from gurobi_solver import GurobiSolver
from OODSelect_utils import load_prediction_data, prepare_dataset, extract_data, probit_transform
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Gurobi solver with grid search')
    parser.add_argument('--N', type=int, required=True,
                        help='Number of OOD samples to select')
    parser.add_argument('--dataset', type=str, default='TerraIncognita',
                        help='Dataset name')
    parser.add_argument('--num_domains', type=int, default=4,
                        help='Number of domains in the dataset')
    parser.add_argument('--train_idxs', type=str, default='0-1-2',
                        help='Training environment indices')
    parser.add_argument('--test_idx', type=str, default='3',
                        help='Test environment index')
    parser.add_argument('--results_dir', type=str, default="",
                        help='Directory containing prediction results')
    parser.add_argument('--time_limit', type=int, default=300,
                        help='Time limit for Gurobi solver (seconds)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Directory to save output selection vector')
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # Load prediction data
    X, y = extract_data(*load_prediction_data(
        args.dataset, args.num_domains,
        args.results_dir, args.train_idxs, args.test_idx
    ))
    train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, \
        X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(X, y)

    # Solve with Gurobi
    solver = GurobiSolver(
        X_train_tensor,
        probit_transform(y_train_tensor),
        time_limit=args.time_limit,
        disp=1
    )
    selected_idxs = solver.solve(args.N)

    # Create selection vector
    selection = torch.zeros((X.shape[1],))
    selection[selected_idxs] = 1

    # Save selection
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        selection,
        os.path.join(args.output_dir, f'Test_{args.dataset}_{args.train_idxs}_{args.test_idx}_{args.N}.pt')
    )

    # Compute correlations
    train_pearson = pearsonr(
        probit_transform(y_train_tensor),
        probit_transform(X_train_tensor[:, selected_idxs].sum(axis=1) / len(selected_idxs))
    )[0]
    test_pearson = pearsonr(
        probit_transform(y_test_tensor),
        probit_transform(X_test_tensor[:, selected_idxs].sum(axis=1) / len(selected_idxs))
    )[0]

    print(f'Train Pearson: {train_pearson}, Test Pearson: {test_pearson}')


if __name__ == "__main__":
    main()
