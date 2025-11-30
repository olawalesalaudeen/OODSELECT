from OODSelect_utils import (
    PearsonRLoss,
    probit_transform,
    load_prediction_data,
    prepare_dataset,
    extract_data,
)
import torch
from gurobi_solver import GurobiSolver
import argparse
import os
from tqdm import trange
import random
random.seed(42)

class GreedySolver:
    """
    Greedy solver for OOD sample selection.

    Starts with a small set from Gurobi solver and greedily adds samples
    that minimize correlation loss.
    """

    def __init__(self, X_train_tensor, y_train, print_progress=True):
        """
        Initialize the greedy solver.

        Args:
            X_train_tensor: Binary correctness matrix of shape (# models, # samples)
            y_train: ID accuracies, shape (# models,). DO NOT apply probit transform.
            print_progress: Whether to print progress updates
        """
        self.X_train_tensor = X_train_tensor
        self.y_train = y_train
        self.print_progress = print_progress
        self.loss_fn = PearsonRLoss()

    def solve(self, max_n=None):
        """
        Solve for selections of increasing sizes using greedy algorithm.

        Args:
            max_n: Maximum number of samples to select (default: all samples)

        Returns:
            Dictionary mapping sample count to binary selection tensor
        """
        if max_n is None:
            max_n = self.X_train_tensor.shape[1]

        selections = {}
        # start with 10 samples from Gurobi solver
        solver = GurobiSolver(self.X_train_tensor, probit_transform(self.y_train), time_limit=1200, disp = self.print_progress)
        selected_idxs = solver.solve(10)
        selection = torch.zeros((self.X_train_tensor.shape[1], ))
        selection[selected_idxs] = 1
        selections[10] = selection.clone()

        unselected_indices = list(set(range(self.X_train_tensor.shape[1])))
        for idx in selected_idxs:
            unselected_indices.remove(idx)

        for i in trange(11, max_n + 1):
            best_loss = float('inf')
            best_idx = None

            random.shuffle(unselected_indices) # shuffle so we don't bias towards the first samples in case of ties
            for j in unselected_indices:
                selection[j] = 1
                loss = self.loss_fn(self.X_train_tensor, self.y_train, selection)
                selection[j] = 0

                if loss < best_loss:
                    best_loss = loss
                    best_idx = j

            selection[best_idx] = 1
            unselected_indices.remove(best_idx)

            if self.print_progress and i % 10 == 0:
                print(f"Selected {i} samples, correlation: {best_loss}")

            selections[i] = selection.clone()

        return selections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run O(MN^2) greedy solver')
    parser.add_argument('--dataset', type=str, default='TerraIncognita')
    parser.add_argument('--num_domains', type=int, default=4)
    parser.add_argument('--train_idxs', type=str, default='0-1-2')
    parser.add_argument('--test_idx', type=str, default='3')
    parser.add_argument('--results_dir', type=str,
                        default="")
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()

    X, y = extract_data(*load_prediction_data(args.dataset, args.num_domains,
                                            args.results_dir, args.train_idxs, args.test_idx))
    train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(X, y)

    solver = GreedySolver(X_train_tensor, y_train_tensor)
    selections = solver.solve(max_n=X.shape[1])
    torch.save(selections, os.path.join(args.output_dir, f'Greedy_{args.dataset}_{args.train_idxs}_{args.test_idx}.pt'))
