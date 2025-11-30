#!/usr/bin/env python
# coding: utf-8

import os
import json
import sys
sys.path.insert(0, '../src/')
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from utils import permute_top_k
import OODSelect_utils
from OODSelect_utils import load_prediction_data, prepare_dataset, extract_data

from datasets import get_dataset_class
from class_names import get_class_names_dict
from scipy.stats import norm

def calculate_correlation_error_bars(correlations, confidence_level=0.95, m_b=None):
    """
    Calculate error bars for correlations using Fisher z-transformation.

    Args:
        correlations: List of correlation values
        confidence_level: Confidence level (default 0.95 for 95% CI)
        m_b: Number of test models (if None, uses length of correlations)

    Returns:
        mean_corr, lower_bound, upper_bound
    """
    if m_b is None:
        m_b = len(correlations)

    # Convert correlations to Fisher z-scores
    z_scores = np.arctanh(np.clip(correlations, -0.999, 0.999))  # Clip to avoid numerical issues

    # Calculate mean z-score
    mean_z = np.mean(z_scores)

    # Calculate confidence interval for z-scores
    alpha = 1 - confidence_level
    z_critical = norm.ppf(1 - alpha/2)
    se_z = z_critical / np.sqrt(m_b - 3)

    # Confidence interval for z-scores
    z_lower = mean_z - se_z
    z_upper = mean_z + se_z

    # Convert back to correlations
    mean_corr = np.tanh(mean_z)
    lower_bound = np.tanh(z_lower)
    upper_bound = np.tanh(z_upper)

    return mean_corr, lower_bound, upper_bound

def parse_args():
    """Parse command line arguments for the correlation plot generation script.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - dataset: Name of the dataset to analyze
            - train_idxs: Indices of domains used for training
            - test_idx: Index of domain used for testing
            - results_dir: Directory containing model results
            - metric: Type of correlation metric to use ('r' or 'r2')
            - training_ood_samples: Number of OOD samples used during training
            - min_samples: Minimum number of samples for correlation plot
            - max_samples: Maximum number of samples for correlation plot
            - num_steps: Number of steps for sample number in correlation plot
            - output_dir: Directory to save output figures
            - explore_selection: Whether to explore selection patterns
            - selection_threshold: Threshold for selection analysis
    """
    parser = argparse.ArgumentParser(description='Generate correlation plots for different OOD sample sizes')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_idxs', type=str, default='0-1-2',
                      help='Training domain indices (default: 0-1-2)')
    parser.add_argument('--test_idx', type=str, default='3',
                      help='Test domain index (default: 3)')
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing prediction results from training')
    parser.add_argument('--selection_dir', type=str, required=True,
                      help='Directory containing OOD selection results')
    parser.add_argument('--metric', type=str, default='r',
                      choices=['r', 'r2', 'spearman'],
                      help='Correlation metric type (default: r)')
    parser.add_argument('--training_ood_samples', type=int, default=None,
                      help='Number of OOD samples used during training (default: 200)')
    parser.add_argument('--min_samples', type=int, default=10,
                      help='Minimum number of samples for correlation plot (default: 10)')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Maximum number of samples for correlation plot (default: 800)')
    parser.add_argument('--max_total_samples', type=int, default=None,
                      help='Maximum number of samples for correlation plot (default: 800)')
    parser.add_argument('--max_total_models', type=int, default=None,
                      help='Maximum number of models for correlation plot (default: 800)')
    parser.add_argument('--seed', type=int, default=0,
                      help='Seed for random number generator (default: 0)')
    parser.add_argument('--num_steps', type=int, default=50,
                      help='Number of steps for sample number in correlation plot (default: 50)')
    parser.add_argument('--output_dir', type=str, default='../',
                      help='Directory to save output figures (default: ../)')
    parser.add_argument('--table_output_dir', type=str, default='../',
                      help='Directory to save output tables (default: ../)')
    parser.add_argument('--random_selection', action='store_true',
                      help='Use random selection instead of soft selection')
    parser.add_argument('--trials', type=int, default=100,
                      help='Number of trials for random selection (default: 100)')
    parser.add_argument('--force', action='store_true',
                      help='Force regeneration of figures even if they already exist')
    parser.add_argument('--split_by_architecture', action='store_true',
                      help='Split models by architecture families instead of randomly')
    return parser.parse_args()


def plot_correlation_vs_samples(num_ood_samples, train_corrs, val_corrs, test_corrs,
                              training_ood_samples, dataset, train_idxs, test_idx, metric,
                              output_dir,
                              random_test_corrs=None,
                              hard_test_corrs=None,
                              force=False):
    """Plot correlation vs number of OOD samples.

    Note: train_corrs and val_corrs now use test models only for consistency.
    """
    """Generate and save a plot showing correlation vs number of OOD samples.

    Args:
        num_ood_samples (list): List of OOD sample sizes used for correlation calculation
        train_corrs (list): Correlation values for training set
        val_corrs (list): Correlation values for validation set
        test_corrs (list): Correlation values for test set
        training_ood_samples (int): Number of OOD samples used during training
        dataset (str): Name of the dataset
        train_idxs (str): Indices of domains used for training
        test_idx (str): Index of domain used for testing
        metric (str): Type of correlation metric used ('r' or 'r2')
        output_dir (str): Directory to save the output figure

    The plot includes:
        - Correlation curves for training, validation, and test sets
        - Vertical line indicating the number of samples used during training
        - Grid lines for better readability
        - Legend identifying each curve
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(num_ood_samples, train_corrs, alpha=0.5, linestyle='--', label='training set', linewidth=4, color=colors[0])
    # ax.plot(num_ood_samples, val_corrs, alpha=0.5, linestyle='--', label='validation set', linewidth=4, color=colors[1])
    # Calculate means and standard errors for plotting (original method)
    test_corrs_mean = [np.mean(x) for x in test_corrs]
    test_corrs_std = [np.std(x) / np.sqrt(len(x)) for x in test_corrs]

    hard_corrs_mean = [np.mean(x) for x in hard_test_corrs] if hard_test_corrs else []
    hard_corrs_std = [np.std(x) / np.sqrt(len(x)) for x in hard_test_corrs] if hard_test_corrs else []

    random_corrs_mean = [np.mean(x) for x in random_test_corrs] if random_test_corrs else []
    random_corrs_std = [np.std(x) / np.sqrt(len(x)) for x in random_test_corrs] if random_test_corrs else []

    # Calculate Fisher z-transformation error bars as additional entry (distance only)
    test_corrs_fisher_err = []
    for x in test_corrs:
        mean_corr, lower, upper = calculate_correlation_error_bars(x)
        test_corrs_fisher_err.append(mean_corr - lower)  # Just the distance

    hard_corrs_fisher_err = []
    if hard_test_corrs:
        for x in hard_test_corrs:
            mean_corr, lower, upper = calculate_correlation_error_bars(x)
            hard_corrs_fisher_err.append(mean_corr - lower)  # Just the distance

    random_corrs_fisher_err = []
    if random_test_corrs:
        for x in random_test_corrs:
            mean_corr, lower, upper = calculate_correlation_error_bars(x)
            random_corrs_fisher_err.append(mean_corr - lower)  # Just the distance

    # Plot main correlation lines
    ax.plot(num_ood_samples, test_corrs_mean, label='test set', linewidth=4, color=colors[2])

    if hard_test_corrs:
        ax.plot(num_ood_samples, hard_corrs_mean, label='hard', linewidth=4, linestyle='-.', alpha=0.75, color=colors[2])
    if random_test_corrs:
        ax.plot(num_ood_samples, random_corrs_mean, label='random', linewidth=4, linestyle=':', alpha=0.75, color=colors[2])

    # Plot error bars (original method)
    ax.fill_between(num_ood_samples,
                   [mean - std for mean, std in zip(test_corrs_mean, test_corrs_std)],
                   [mean + std for mean, std in zip(test_corrs_mean, test_corrs_std)],
                   alpha=0.2, color=colors[2], label='test set error bars')

    if hard_test_corrs:
        ax.fill_between(num_ood_samples,
                       [mean - std for mean, std in zip(hard_corrs_mean, hard_corrs_std)],
                       [mean + std for mean, std in zip(hard_corrs_mean, hard_corrs_std)],
                       alpha=0.2, color=colors[2])

    if random_test_corrs:
        ax.fill_between(num_ood_samples,
                       [mean - std for mean, std in zip(random_corrs_mean, random_corrs_std)],
                       [mean + std for mean, std in zip(random_corrs_mean, random_corrs_std)],
                       alpha=0.2, color=colors[2])

    ax.set_xlabel('N OOD samples', fontsize=24)
    ax.set_xscale('log')
    if metric == 'r':
        ax.set_ylabel('Pearson R', fontsize=24)
    elif metric == 'r2':
        ax.set_ylabel('Pearson R^2', fontsize=24)
    elif metric == 'spearman':
        ax.set_ylabel('Spearman R', fontsize=24)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    ax.grid(True)
    ax.legend(fontsize=20)

    fig.tight_layout()
    output_path = os.path.join(
        output_dir,
        f"{dataset}_{train_idxs}_{test_idx}{'-'+str(training_ood_samples) if training_ood_samples is not None else ''}_{metric}.pdf"
    )

    # Check if file already exists and skip if not forcing
    if not force and os.path.exists(output_path):
        print(f"Figure already exists: {output_path}. Use --force to regenerate.")
        plt.close()
        return {
            'n': num_ood_samples,
            'train_corrs': train_corrs,
            'val_corrs': val_corrs,
            'test_corrs': test_corrs_mean,
            'random_test_corrs': random_corrs_mean,
            'hard_test_corrs': hard_corrs_mean,
            'test_corrs_stderr': test_corrs_std,
            'random_test_corrs_stderr': random_corrs_std,
            'hard_test_corrs_stderr': hard_corrs_std,
            'test_corrs_fisher_err': test_corrs_fisher_err,
            'random_test_corrs_fisher_err': random_corrs_fisher_err,
            'hard_test_corrs_fisher_err': hard_corrs_fisher_err,
        }

    plt.savefig(output_path)
    print(f"Saved figure: {output_path}")
    plt.close()

    return {
        'n': num_ood_samples,
        'train_corrs': train_corrs,
        'val_corrs': val_corrs,
        'test_corrs': test_corrs_mean,
        'random_test_corrs': random_corrs_mean,
        'hard_test_corrs': hard_corrs_mean,
        'test_corrs_stderr': test_corrs_std,
        'random_test_corrs_stderr': random_corrs_std,
        'hard_test_corrs_stderr': hard_corrs_std,
        'test_corrs_fisher_err': test_corrs_fisher_err,
        'random_test_corrs_fisher_err': random_corrs_fisher_err,
        'hard_test_corrs_fisher_err': hard_corrs_fisher_err,
    }


def main():
    """Main execution function for generating correlation plots.

    This function:
    1. Parses command line arguments
    2. Loads and prepares the dataset
    3. Loads the trained model's selection weights
    4. Calculates correlations for different sample sizes
    5. Generates and saves the correlation plot
    6. Optionally explores selection patterns
    """
    # Parse command line arguments
    args = parse_args()

    gen = torch.Generator().manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.table_output_dir, exist_ok=True)

    num_domains = len(get_dataset_class(args.dataset).ENVIRONMENTS)
    # Load predictions and configuration
    if args.split_by_architecture:
        predictions_data, acc_cols, model_architectures = load_prediction_data(args.dataset, num_domains, args.results_dir,
                                  args.train_idxs, args.test_idx, args.max_total_samples, args.max_total_models,
                                  seed=args.seed, extract_architectures=args.split_by_architecture)
    else:
        predictions_data, acc_cols = load_prediction_data(args.dataset, num_domains, args.results_dir,
                                  args.train_idxs, args.test_idx, args.max_total_samples, args.max_total_models,
                                  seed=args.seed, extract_architectures=args.split_by_architecture)
    X, y = extract_data(predictions_data, acc_cols)

    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")

    # Prepare training, validation, and test datasets
    if args.split_by_architecture:
        train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, \
        X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(X, y,
                                                                                   split_by_architecture=True,
                                                                                   model_architectures=model_architectures,
                                                                                   seed=args.seed)
    else:
        train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, \
        X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(X, y, seed=args.seed)

    # Get number of OOD samples
    max_num_ood_samples = X_train_tensor.shape[1]
    assert max_num_ood_samples == X_test_tensor.shape[1]

    if args.training_ood_samples is None:
        training_num_ood_samples_list = [int(f.split('_')[-2]) for f in os.listdir(args.selection_dir) if f.startswith(f"Test_{args.dataset}_{args.train_idxs}_{args.test_idx}_") and int(f.split('_')[-2]) <= max_num_ood_samples]
    else:
        training_num_ood_samples_list = [args.training_ood_samples]


    results = {}
    removed_training_num_ood_samples = []
    print(training_num_ood_samples_list)
    # Load selections for each training_num_ood_samples
    for training_num_ood_samples in training_num_ood_samples_list:
        # Load Soft Selection
        if args.random_selection:
            selection = torch.rand(max_num_ood_samples, generator=gen).to(X_test_tensor.device)
        else:
            try:
                selection_path = os.path.join(
                    args.selection_dir,
                    f"Test_{args.dataset}_{args.train_idxs}_{args.test_idx}_{training_num_ood_samples}_r",
                    "best_model_state_r.pt"
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    selection = torch.tensor(torch.load(selection_path, weights_only=False))#.to(X_test_tensor.device)
            except FileNotFoundError:
                print(f"File not found: {selection_path}")
                removed_training_num_ood_samples.append(training_num_ood_samples)
                continue

        if selection.shape[0] > max_num_ood_samples:
            print(f"Selection shape is greater than max_num_ood_samples: {selection.shape[0]} > {max_num_ood_samples}")
            removed_training_num_ood_samples.append(training_num_ood_samples)
            continue
        results[training_num_ood_samples] = {'selection': selection}

    results[max_num_ood_samples] = {'selection': torch.ones(max_num_ood_samples)}

    print(f"selection.shape: {selection.shape}")
    training_num_ood_samples_list = [n for n in training_num_ood_samples_list if n not in removed_training_num_ood_samples]

    if args.max_samples is None:
        args.max_samples = max_num_ood_samples

    if args.training_ood_samples is None:
        sample_sizes = sorted(list(results.keys()))
    else:
        sample_sizes = np.linspace(args.min_samples, args.max_samples, args.num_steps, dtype=int)

    # Calculate correlations for different sample sizes
    num_ood_samples_list, train_corrs, val_corrs, test_corrs = [], [], [], []
    random_test_corrs = []
    hard_test_corrs = []

    if args.training_ood_samples is not None:
        assert len(results) == 1

    print("sample_sizes:", sample_sizes)

    # Find the next larger sample size for each n to create error bars
    def find_next_larger_sample_size(n, sample_sizes):
        """Find the next larger sample size in the list."""
        larger_sizes = [s for s in sample_sizes if s > n]
        return min(larger_sizes) if larger_sizes else None

    for i, n in enumerate(sample_sizes):
        # Find the next larger sample size for error bar calculation
        next_m = find_next_larger_sample_size(n, sample_sizes)
        print(f"Sample size {n}: using {next_m} for error bars" if next_m else f"Sample size {n}: no larger size available for error bars")

        for mode in ['selection', 'random', 'hard']:
            if args.training_ood_samples is None:
                selection = results[n]['selection']
            else:
                selection = results[args.training_ood_samples]['selection']

            if mode == 'random':
                selection = torch.rand(selection.shape, generator=gen).to(X_test_tensor.device)
            elif mode == 'hard':
                accs = X_test_tensor.mean(0)
                selection = 1. - accs

            if mode == 'selection':
                # Calculate main correlation and error bars by subsampling from next larger size
                if next_m is not None and next_m in results:
                    # Use the selection from the larger sample size for both main correlation and error bars
                    larger_selection = results[next_m]['selection']
                    # Subsample n from m multiple times to create error bars
                    perm_selections = permute_top_k(larger_selection, n, args.trials)

                    # Calculate correlations for all subsamples (including main)
                    all_corrs = []
                    for perm_selection in perm_selections:
                        all_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor,
                                                            perm_selection, n, args.metric))

                    # Use the first subsample as the main correlation
                    main_corr = all_corrs[0]
                    test_corrs.append(all_corrs)

                    # Calculate train and val correlations using the first subsample (using test models only)
                    train_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, perm_selections[0], n, args.metric))
                    val_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, perm_selections[0], n, args.metric))
                else:
                    # Fallback: use original selection for size n (using test models only)
                    main_corr = OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, selection, n, args.metric)
                    train_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, selection, n, args.metric))
                    val_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, selection, n, args.metric))
                    test_corrs.append([main_corr])

            elif mode == 'random':
                # Calculate main correlation and error bars by subsampling from next larger size
                if next_m is not None and next_m in results:
                    # Use random selection from the larger sample size
                    larger_selection = torch.rand(results[next_m]['selection'].shape, generator=gen).to(X_test_tensor.device)
                    perm_selections = permute_top_k(larger_selection, n, args.trials)

                    # Calculate correlations for all subsamples (including main)
                    all_corrs = []
                    for perm_selection in perm_selections:
                        all_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor,
                                                            perm_selection, n, args.metric))

                    random_test_corrs.append(all_corrs)
                else:
                    # Fallback: use original selection for size n
                    main_corr = OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, selection, n, args.metric)
                    random_test_corrs.append([main_corr])

            elif mode == 'hard':
                # Calculate main correlation and error bars by subsampling from next larger size
                if next_m is not None and next_m in results:
                    # Use hard selection from the larger sample size
                    larger_accs = X_test_tensor.mean(0)
                    larger_selection = 1. - larger_accs
                    perm_selections = permute_top_k(larger_selection, n, args.trials)

                    # Calculate correlations for all subsamples (including main)
                    all_corrs = []
                    for perm_selection in perm_selections:
                        all_corrs.append(OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor,
                                                            perm_selection, n, args.metric))

                    hard_test_corrs.append(all_corrs)
                else:
                    # Fallback: use original selection for size n
                    main_corr = OODSelect_utils.TestSetFinder.test(X_test_tensor, y_test_tensor, selection, n, args.metric)
                    hard_test_corrs.append([main_corr])

        num_ood_samples_list.append(n)

    # Plot results
    corr_dict = plot_correlation_vs_samples(
        num_ood_samples_list, train_corrs, val_corrs, test_corrs,
        args.training_ood_samples, args.dataset, args.train_idxs, args.test_idx,
        args.metric, args.output_dir,
        random_test_corrs, hard_test_corrs,
        force=args.force
    )

    with open(os.path.join(args.table_output_dir, f"corr_dict_{args.dataset}_{args.train_idxs}_{args.test_idx}{'-'+str(args.training_ood_samples) if args.training_ood_samples is not None else ''}_{args.metric}.json"), "w") as f:
        json.dump(corr_dict, f)

    # Explore selection patterns
    value_range = np.arange(get_dataset_class(args.dataset).NUM_CLASSES)  # class indices 0–9
    class_percentages_across_samples = []
    random_class_percentages_across_samples = []
    hard_class_percentages_across_samples = []

    for n in sample_sizes:
        if args.training_ood_samples is None:
            selection = results[n]['selection']
        else:
            selection = results[args.training_ood_samples]['selection']

        ordered_indices_desc = np.argsort(-selection.cpu().numpy())

        hard_selection = torch.tensor((1. - X.mean(0)).astype(np.float32))
        random_selection = torch.rand(selection.shape, generator=gen).to(X_test_tensor.device)

        hard_ordered_indices_desc = np.argsort(-hard_selection.cpu().numpy())
        random_ordered_indices_desc = np.argsort(-random_selection.cpu().numpy())

        # Loop through different numbers of selected samples
        ys = predictions_data[0][-1].iloc[ordered_indices_desc[:n]].label.values
        counts = np.array([np.sum(ys == val) for val in value_range])
        percentages = 100 * counts / counts.sum()
        class_percentages_across_samples.append(percentages)

        hard_ys = predictions_data[0][-1].iloc[hard_ordered_indices_desc[:n]].label.values
        hard_counts = np.array([np.sum(hard_ys == val) for val in value_range])
        hard_percentages = 100 * hard_counts / hard_counts.sum()
        hard_class_percentages_across_samples.append(hard_percentages)

        random_ys = predictions_data[0][-1].iloc[random_ordered_indices_desc[:n]].label.values
        random_counts = np.array([np.sum(random_ys == val) for val in value_range])
        random_percentages = 100 * random_counts / random_counts.sum()
        random_class_percentages_across_samples.append(random_percentages)

    # Convert to a numpy array for easier plotting
    class_percentages_across_samples = np.array(class_percentages_across_samples)
    hard_class_percentages_across_samples = np.array(hard_class_percentages_across_samples)
    random_class_percentages_across_samples = np.array(random_class_percentages_across_samples)

    # Get class names for labeling
    class_labels = [get_class_names_dict(args.dataset)[i] for i in value_range]
    labels = [get_class_names_dict(args.dataset)[i] for i in value_range]

    # If there are more than 10 classes, select top 5 and bottom 5 based on smallest n
    if len(value_range) > 6:
        print(f"len(value_range) > 6: {len(value_range)}; reducing to 6: 3 min and 3 max")
        # Get percentages at smallest n
        smallest_n_percentages = class_percentages_across_samples[0]
        # Sort indices by percentage
        sorted_indices = np.argsort(smallest_n_percentages)
        # Get top 5 and bottom 5 indices
        selected_indices = np.concatenate([sorted_indices[-3:], sorted_indices[:3]])
        # Filter the data
        class_percentages_across_samples = class_percentages_across_samples[:, selected_indices]
        hard_class_percentages_across_samples = hard_class_percentages_across_samples[:, selected_indices]
        random_class_percentages_across_samples = random_class_percentages_across_samples[:, selected_indices]

        labels = list(set([labels[i] for i in selected_indices]))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, label in enumerate(labels):
        print(i, class_percentages_across_samples.shape)
        ax.plot(sample_sizes, class_percentages_across_samples[:, i],
                 label=label, linewidth=2, color=colors[class_labels.index(label)])

        ax.plot(sample_sizes, hard_class_percentages_across_samples[:, i],
                 linewidth=2, color=colors[class_labels.index(label)], linestyle='-.', alpha=0.5)

        ax.plot(sample_sizes, random_class_percentages_across_samples[:, i],
                 linewidth=2, color=colors[class_labels.index(label)], linestyle=':', alpha=0.5)

    ax.plot([], [], linestyle='-.', color='k', label='hard')
    ax.plot([], [], linestyle=':', color='k', label='random')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel("N OODSelect Samples", fontsize=24)
    ax.set_ylabel("Percent Included (%)", fontsize=24)
    ax.set_xscale('log')

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=20)  # moves legend to the right, centered vertically
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()


    output_path = os.path.join(
        args.output_dir,
        f"class_freq_{args.metric}_{args.dataset}_{args.train_idxs}_{args.test_idx}{'-'+str(args.training_ood_samples) if args.training_ood_samples is not None else ''}_{args.metric}.pdf"
    )
    plt.savefig(output_path)

    plt.close()

    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()

