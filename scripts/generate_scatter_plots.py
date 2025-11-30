#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, '../src/')
from OODSelect_utils import load_prediction_data, prepare_dataset, extract_data
from datasets import get_dataset_class
from scipy.stats import norm, linregress

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--num_oods', nargs='+', type=int, default=[250])
parser.add_argument('--train_idxs', type=str, default='0-1-2')
parser.add_argument('--test_idx', type=int, default=3)
parser.add_argument('--max_total_samples', type=int, default=None)
parser.add_argument('--max_total_models', type=int, default=None)
parser.add_argument('--results_dir', type=str, required=True,
                   help='Directory containing prediction results from training')
parser.add_argument('--save_dir', type=str, default='./results_figures',
                   help='Directory to save scatter plots')
parser.add_argument('--selection_base', type=str, required=True,
                   help='Base directory containing OOD selection results')
parser.add_argument('--force', action='store_true',
                   help='Force regeneration of figures even if they already exist')
parser.add_argument('--metric', type=str, default='r',
                   choices=['r', 'r2', 'spearman'],
                   help='Correlation metric type (default: r)')
parser.add_argument('--split_by_architecture', action='store_true',
                   help='Split models by architecture families instead of randomly')
parser.add_argument('--seed', type=int, default=0,
                   help='Random seed for splitting (default: 0)')

args = parser.parse_args()

def save_figure_if_needed(fig, filepath, force=False):
    """Save figure only if it doesn't exist or force is True."""
    if not force and os.path.exists(filepath):
        print(f"Figure already exists: {filepath}. Use --force to regenerate.")
        plt.close(fig)
        return False
    else:
        plt.savefig(filepath)
        print(f"Saved figure: {filepath}")
        plt.close(fig)
        return True

results_dir = args.results_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

dataset = args.dataset
num_domains = len(get_dataset_class(dataset).ENVIRONMENTS)
train_idxs = args.train_idxs
test_envs = [args.test_idx]

res_df_dict = {}
for test_env in test_envs:
    if args.split_by_architecture:
        predictions_data, acc_cols, model_architectures = load_prediction_data(dataset, num_domains, results_dir,
                              train_idxs, str(test_env), extract_architectures=args.split_by_architecture)
    else:
        predictions_data, acc_cols = load_prediction_data(dataset, num_domains, results_dir,
                              train_idxs, str(test_env), extract_architectures=args.split_by_architecture)
        model_architectures = None
    X, y = extract_data(predictions_data, acc_cols)

    args.max_total_samples = args.max_total_samples or X.shape[1]
    args.max_total_models = args.max_total_models or X.shape[0]

    X = X[:args.max_total_models, :args.max_total_samples]
    y = y[:args.max_total_models]

    train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, \
    X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor = prepare_dataset(
        X, y,
        split_by_architecture=args.split_by_architecture,
        model_architectures=model_architectures if args.split_by_architecture else None,
        seed=args.seed
    )

    # Only use test models for scatter plots
    X = X_test_tensor
    y = y_test_tensor
    train_accs = y.detach().numpy()
    test_accs = X.detach().numpy().mean(1)

    test_dfs = {}
    for num_ood in args.num_oods:
        selection_path = f'{args.selection_base}/Test_{dataset}_{train_idxs}_{str(test_env)}_{num_ood}_r/best_model_state_r.pt'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                selection = torch.tensor(torch.load(selection_path, weights_only=False))
            sorted_ood_indices = torch.topk(selection, num_ood).indices.detach().numpy()
        except FileNotFoundError:
            print(f'{selection_path} not found')
            continue
        except Exception as e:
            print(f'Error loading {selection_path}: {e}')
            continue

        selected_test = X.detach().numpy()[:, sorted_ood_indices]
        selected_test_accs = selected_test.mean(1)

        df = pd.DataFrame([train_accs, test_accs, selected_test_accs], index=['train_acc', 'test_acc', 'selected_test_acc']).T
        test_dfs[num_ood] = df

    res_df_dict[test_env] = test_dfs

n_bins = 25
for test_env, test_res_df in res_df_dict.items():
    for num_ood, res_df in test_res_df.items():
        # Recalculate sorted_ood_indices for this specific num_ood
        selection_path = f'{args.selection_base}/Test_{dataset}_{train_idxs}_{str(test_env)}_{num_ood}_r/best_model_state_r.pt'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                selection = torch.tensor(torch.load(selection_path, weights_only=False))
            sorted_ood_indices = torch.topk(selection, num_ood).indices.detach().numpy()
        except (FileNotFoundError, Exception) as e:
            print(f'Error loading {selection_path}: {e}')
            continue

        fig, axes = plt.subplots(1, 3, figsize=(24,5))

        eps = 1e-6
        res_df['train_acc_clipped'] = np.clip(res_df['train_acc'], eps, 1 - eps)
        res_df['test_acc_clipped']  = np.clip(res_df['test_acc'],  eps, 1 - eps)
        res_df['selected_test_acc_clipped']  = np.clip(res_df['selected_test_acc'],  eps, 1 - eps)

        res_df['train_acc_probit'] = norm.ppf(res_df['train_acc_clipped'])
        res_df['test_acc_probit']  = norm.ppf(res_df['test_acc_clipped'])
        res_df['selected_test_acc_probit']  = norm.ppf(res_df['selected_test_acc_clipped'])

        sns.scatterplot(data=res_df, x='train_acc_probit', y='selected_test_acc_probit', ax=axes[0])

        # Use the same correlation calculation method as JSON files
        from OODSelect_utils import TestSetFinder, probit_transform

        # Calculate correlation for "All" data using TestSetFinder method
        X_tensor = torch.tensor(X.detach().numpy(), dtype=torch.float32)
        y_tensor = torch.tensor(y.detach().numpy(), dtype=torch.float32)

        # For "All" correlation, we need to create a selection that selects all samples
        all_selection = torch.ones(X_tensor.shape[1])
        r_value = TestSetFinder.test(X_tensor, y_tensor, all_selection, X_tensor.shape[1], args.metric, probit_transform)

        # For "OODSelect" correlation, use the actual selection weights
        selected_r_value = TestSetFinder.test(X_tensor, y_tensor, selection, num_ood, args.metric, probit_transform)

        # For plotting, we still need the regression lines, so calculate them separately
        X_probit = res_df['train_acc_probit']
        Y_probit = res_df['test_acc_probit']
        slope, intercept, _, _, _ = linregress(X_probit, Y_probit)

        selected_X = res_df['train_acc_probit']
        selected_Y = res_df['selected_test_acc_probit']
        selected_slope, selected_intercept, _, _, _ = linregress(selected_X, selected_Y)

        # Set x-axis range to span the full range of all data points in probit space
        # Use the combined range of both datasets
        all_x_values = np.concatenate([X_probit, selected_X])
        x_min_probit = all_x_values.min()
        x_max_probit = all_x_values.max()
        x_range_probit = x_max_probit - x_min_probit
        x_min_extended = x_min_probit - 0.1 * x_range_probit  # Increased padding
        x_max_extended = x_max_probit + 0.1 * x_range_probit  # Increased padding

        xx = np.linspace(x_min_extended, x_max_extended, 200)
        yy = slope * xx + intercept
        axes[0].plot(xx, yy, linestyle=':', linewidth=2, color='k', label='All')

        selected_xx = np.linspace(x_min_extended, x_max_extended, 200)
        selected_yy = selected_slope * selected_xx + selected_intercept
        axes[0].plot(selected_xx, selected_yy, linewidth=2, color='r', label='OODSelect')

        # Set the x-axis limits to ensure full range
        axes[0].set_xlim(x_min_extended, x_max_extended)

        annotation_text = (
            "OODSelect (All)\n"
            f"Pearson R = {selected_r_value:.2f} ({r_value:.2f})\n"
            f"Slope = {selected_slope:.2f} ({slope:.2f})\n"
            f"Intercept = {selected_intercept:.2f} ({intercept:.2f})"
        )
        axes[0].text(
            0.05, 0.95,
            annotation_text,
            transform=axes[0].transAxes,
            va='top',
            fontsize=18,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        n_ticks = 6

        x_min = res_df['train_acc_clipped'].min()
        x_max = res_df['train_acc_clipped'].max()
        x_ticks_frac = np.linspace(x_min, x_max, n_ticks)
        x_ticks_probit = norm.ppf(x_ticks_frac)
        axes[0].set_xticks(x_ticks_probit)
        axes[0].set_xticklabels([f"{v:.2g}" for v in x_ticks_frac], rotation=90)

        y_min = res_df['test_acc_clipped'].min()
        y_max = res_df['test_acc_clipped'].max()
        y_ticks_frac = np.linspace(y_min, y_max, n_ticks)
        y_ticks_probit = norm.ppf(y_ticks_frac)
        axes[0].set_yticks(y_ticks_probit)
        axes[0].set_yticklabels([f"{v:.2g}" for v in y_ticks_frac])

        axes[0].tick_params(axis='x', labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[0].set_xlabel("ID Acc", fontsize=24)
        axes[0].set_ylabel("OOD Acc", fontsize=24)
        axes[0].grid(True)
        axes[0].legend(fontsize=20, loc='lower left')

        sns.histplot(data=res_df, x='train_acc', ax=axes[1], bins=n_bins)
        axes[1].tick_params(axis='x', labelsize=18)
        axes[1].tick_params(axis='y', labelsize=18)
        axes[1].set_xlabel("ID Acc", fontsize=24)
        axes[1].set_ylabel("Count", fontsize=24)
        axes[1].set_yscale('log')

        res_df_ = res_df[['train_acc', 'test_acc', 'selected_test_acc']]
        res_df_.columns = ['ID Acc', 'All', 'OODSelect']
        plot_df = pd.melt(
            res_df_,
            value_vars=['All', 'OODSelect'],
            var_name='Selection',
            value_name='Accuracy'
        )
        plot_df = plot_df.sort_values(by=['Selection'])
        sns.histplot(
            data=plot_df,
            x='Accuracy',
            hue='Selection',
            ax=axes[2],
            bins=n_bins,
        )
        axes[2].tick_params(axis='x', labelsize=18)
        axes[2].tick_params(axis='y', labelsize=18)
        axes[2].set_xlabel("OOD Acc", fontsize=24)
        axes[2].set_ylabel("Count", fontsize=24)
        axes[2].set_yscale('log')
        axes[2].legend(labels=['OODSelect', 'All'], fontsize=20)

        # Add more padding to prevent label cutoff
        fig.tight_layout(pad=2.0)
        filepath = os.path.join(save_dir, f'{dataset}_model_accs_{train_idxs}_{test_env}_selected_{num_ood}_{args.metric}.pdf')
        save_figure_if_needed(fig, filepath, args.force)

        fig, ax = plt.subplots(figsize=(8,5))
        axes = [ax]

        eps = 1e-6
        res_df['train_acc_clipped'] = np.clip(res_df['train_acc'], eps, 1 - eps)
        res_df['test_acc_clipped']  = np.clip(res_df['test_acc'],  eps, 1 - eps)

        res_df['train_acc_probit'] = norm.ppf(res_df['train_acc_clipped'])
        res_df['selected_test_acc_probit']  = norm.ppf(res_df['selected_test_acc_clipped'])

        sns.scatterplot(data=res_df, x='train_acc_probit', y='selected_test_acc_probit', ax=axes[0])

        X_probit = res_df['train_acc_probit']
        Y_probit = res_df['selected_test_acc_probit']
        selected_slope, selected_intercept, selected_r_value, selected_p_value, selected_std_err = linregress(selected_X, selected_Y)

        # Set x-axis range to span the full range of all data points in probit space
        x_min_probit = selected_X.min()
        x_max_probit = selected_X.max()
        x_range_probit = x_max_probit - x_min_probit
        x_min_extended = x_min_probit - 0.1 * x_range_probit  # Increased padding
        x_max_extended = x_max_probit + 0.1 * x_range_probit  # Increased padding

        selected_xx = np.linspace(x_min_extended, x_max_extended, 200)
        selected_yy = selected_slope * selected_xx + selected_intercept
        axes[0].plot(selected_xx, selected_yy, linewidth=2, color='k')

        # Set the x-axis limits to ensure full range
        axes[0].set_xlim(x_min_extended, x_max_extended)

        annotation_text = (
            f"Pearson R = {selected_r_value:.2f}\n"
            f"Slope = {selected_slope:.2f}\n"
            f"Intercept = {selected_intercept:.2f}"
        )
        axes[0].text(
            0.05, 0.95,
            annotation_text,
            transform=axes[0].transAxes,
            va='top',
            fontsize=20,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        n_ticks = 6

        x_min = res_df['train_acc_clipped'].min()
        x_max = res_df['train_acc_clipped'].max()
        x_ticks_frac = np.linspace(x_min, x_max, n_ticks)
        x_ticks_probit = norm.ppf(x_ticks_frac)
        axes[0].set_xticks(x_ticks_probit)
        axes[0].set_xticklabels([f"{v:.2g}" for v in x_ticks_frac], rotation=90)

        y_min = res_df['selected_test_acc_clipped'].min()
        y_max = res_df['selected_test_acc_clipped'].max()
        y_ticks_frac = np.linspace(y_min, y_max, n_ticks)
        y_ticks_probit = norm.ppf(y_ticks_frac)
        axes[0].set_yticks(y_ticks_probit)
        axes[0].set_yticklabels([f"{v:.2g}" for v in y_ticks_frac])

        axes[0].tick_params(axis='x', labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[0].set_xlabel("ID Acc", fontsize=24)
        axes[0].set_ylabel("OODSelect Acc", fontsize=24)
        axes[0].grid(True)

        fig.tight_layout()
        filepath = os.path.join(save_dir, f'{dataset}_model_accs_aotl_{train_idxs}_{test_env}_selected_{num_ood}_{args.metric}.pdf')
        save_figure_if_needed(fig, filepath, args.force)
        fig, ax = plt.subplots(figsize=(8,5))
        axes = [ax]

        eps = 1e-6
        res_df['train_acc_clipped'] = np.clip(res_df['train_acc'], eps, 1 - eps)
        res_df['test_acc_clipped']  = np.clip(res_df['test_acc'],  eps, 1 - eps)

        res_df['train_acc_probit'] = norm.ppf(res_df['train_acc_clipped'])
        res_df['test_acc_probit']  = norm.ppf(res_df['test_acc_clipped'])

        sns.scatterplot(data=res_df, x='train_acc_probit', y='test_acc_probit', ax=axes[0])

        X_probit = res_df['train_acc_probit']
        Y_probit = res_df['test_acc_probit']
        slope, intercept, r_value, p_value, std_err = linregress(X_probit, Y_probit)

        # Set x-axis range to span the full range of all data points in probit space
        x_min_probit = X_probit.min()
        x_max_probit = X_probit.max()
        x_range_probit = x_max_probit - x_min_probit
        x_min_extended = x_min_probit - 0.1 * x_range_probit  # Increased padding
        x_max_extended = x_max_probit + 0.1 * x_range_probit  # Increased padding

        xx = np.linspace(x_min_extended, x_max_extended, 200)
        yy = slope * xx + intercept
        axes[0].plot(xx, yy, linewidth=2, color='k')

        # Set the x-axis limits to ensure full range
        axes[0].set_xlim(x_min_extended, x_max_extended)

        annotation_text = (
            f"Pearson R = {r_value:.2f}\n"
            f"Slope = {slope:.2f}\n"
            f"Intercept = {intercept:.2f}"
        )
        axes[0].text(
            0.05, 0.95,
            annotation_text,
            transform=axes[0].transAxes,
            va='top',
            fontsize=20,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        n_ticks = 6

        x_min = res_df['train_acc_clipped'].min()
        x_max = res_df['train_acc_clipped'].max()
        x_ticks_frac = np.linspace(x_min, x_max, n_ticks)
        x_ticks_probit = norm.ppf(x_ticks_frac)
        axes[0].set_xticks(x_ticks_probit)
        axes[0].set_xticklabels([f"{v:.2g}" for v in x_ticks_frac], rotation=90)

        y_min = res_df['test_acc_clipped'].min()
        y_max = res_df['test_acc_clipped'].max()
        y_ticks_frac = np.linspace(y_min, y_max, n_ticks)
        y_ticks_probit = norm.ppf(y_ticks_frac)
        axes[0].set_yticks(y_ticks_probit)
        axes[0].set_yticklabels([f"{v:.2g}" for v in y_ticks_frac])

        axes[0].tick_params(axis='x', labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[0].set_xlabel("ID Acc", fontsize=24)
        axes[0].set_ylabel("OOD Acc", fontsize=24)
        axes[0].grid(True)

        fig.tight_layout()
        filepath = os.path.join(save_dir, f'{dataset}_model_accs_aotl_{train_idxs}_{test_env}_{args.metric}.pdf')
        save_figure_if_needed(fig, filepath, args.force)
        fig, ax = plt.subplots(figsize=(8,5))
        axes = [ax]
        colors = list(sns.color_palette("colorblind"))

        eps = 1e-6
        res_df['train_acc_clipped'] = np.clip(res_df['train_acc'], eps, 1 - eps)
        res_df['test_acc_clipped']  = np.clip(res_df['test_acc'],  eps, 1 - eps)
        res_df['selected_test_acc_clipped']  = np.clip(res_df['selected_test_acc'],  eps, 1 - eps)

        res_df['train_acc_probit'] = norm.ppf(res_df['train_acc_clipped'])
        res_df['test_acc_probit']  = norm.ppf(res_df['test_acc_clipped'])
        res_df['selected_test_acc_probit']  = norm.ppf(res_df['selected_test_acc_clipped'])

        sns.scatterplot(data=res_df, x='train_acc_probit', y='selected_test_acc_probit', ax=axes[0], color=colors[0])
        sns.scatterplot(data=res_df, x='train_acc_probit', y='test_acc_probit', ax=axes[0], color=colors[1])

        X_probit = res_df['train_acc_probit']
        Y_probit = res_df['test_acc_probit']
        slope, intercept, r_value, p_value, std_err = linregress(X_probit, Y_probit)

        selected_X = res_df['train_acc_probit']
        selected_Y = res_df['selected_test_acc_probit']
        selected_slope, selected_intercept, selected_r_value, selected_p_value, selected_std_err = linregress(selected_X, selected_Y)

        # Set x-axis range to span the full range of all data points in probit space
        # Use the combined range of both datasets
        all_x_values = np.concatenate([X_probit, selected_X])
        x_min_probit = all_x_values.min()
        x_max_probit = all_x_values.max()
        x_range_probit = x_max_probit - x_min_probit
        x_min_extended = x_min_probit - 0.1 * x_range_probit  # Increased padding
        x_max_extended = x_max_probit + 0.1 * x_range_probit  # Increased padding

        xx = np.linspace(x_min_extended, x_max_extended, 200)
        yy = slope * xx + intercept
        axes[0].plot(xx, yy, linestyle='-', linewidth=2, label='All', color=colors[1])

        selected_xx = np.linspace(x_min_extended, x_max_extended, 200)
        selected_yy = selected_slope * selected_xx + selected_intercept
        axes[0].plot(selected_xx, selected_yy, linewidth=2, label='OODSelect', color=colors[0])

        # Set the x-axis limits to ensure full range
        axes[0].set_xlim(x_min_extended, x_max_extended)

        annotation_text = (
            "OODSelect (All)\n"
            f"Pearson R = {selected_r_value:.2f} ({r_value:.2f})\n"
            f"Slope = {selected_slope:.2f} ({slope:.2f})\n"
            f"Intercept = {selected_intercept:.2f} ({intercept:.2f})"
        )
        axes[0].text(
            0.05, 0.95,
            annotation_text,
            transform=axes[0].transAxes,
            va='top',
            fontsize=18,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        n_ticks = 6

        x_min = res_df['train_acc_clipped'].min()
        x_max = res_df['train_acc_clipped'].max()
        x_ticks_frac = np.linspace(x_min, x_max, n_ticks)
        x_ticks_probit = norm.ppf(x_ticks_frac)
        axes[0].set_xticks(x_ticks_probit)
        axes[0].set_xticklabels([f"{v:.2g}" for v in x_ticks_frac], rotation=90)

        y_min = res_df['test_acc_clipped'].min()
        y_max = res_df['test_acc_clipped'].max()
        y_ticks_frac = np.linspace(y_min, y_max, n_ticks)
        y_ticks_probit = norm.ppf(y_ticks_frac)
        axes[0].set_yticks(y_ticks_probit)
        axes[0].set_yticklabels([f"{v:.2g}" for v in y_ticks_frac])

        axes[0].tick_params(axis='x', labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[0].set_xlabel("ID Acc", fontsize=24)
        axes[0].set_ylabel("OOD Acc", fontsize=24)
        axes[0].grid(True)
        axes[0].legend(fontsize=20, loc='lower left')

        # Add more padding to prevent label cutoff
        fig.tight_layout(pad=2.0)
        filepath = os.path.join(save_dir, f'{dataset}_model_accs_aotl_{train_idxs}_{test_env}_both_{num_ood}_{args.metric}.pdf')
        save_figure_if_needed(fig, filepath, args.force)

        # NEW: Three-way comparison: All, Selected, and Not Selected
        fig, ax = plt.subplots(figsize=(8,5))
        axes = [ax]
        colors = list(sns.color_palette("colorblind"))

        eps = 1e-6
        res_df['train_acc_clipped'] = np.clip(res_df['train_acc'], eps, 1 - eps)
        res_df['test_acc_clipped']  = np.clip(res_df['test_acc'],  eps, 1 - eps)
        res_df['selected_test_acc_clipped']  = np.clip(res_df['selected_test_acc'],  eps, 1 - eps)

        res_df['train_acc_probit'] = norm.ppf(res_df['train_acc_clipped'])
        res_df['test_acc_probit']  = norm.ppf(res_df['test_acc_clipped'])
        res_df['selected_test_acc_probit']  = norm.ppf(res_df['selected_test_acc_clipped'])

        # Create not selected data
        all_indices = set(range(X.shape[1]))  # Use number of samples, not models
        selected_indices = set(sorted_ood_indices)
        not_selected_indices = list(all_indices - selected_indices)

        # Calculate not selected test accuracies
        not_selected_test = X.detach().numpy()[:, not_selected_indices]
        not_selected_test_accs = not_selected_test.mean(1)

        # Add not selected to dataframe
        res_df['not_selected_test_acc'] = not_selected_test_accs
        res_df['not_selected_test_acc_clipped'] = np.clip(res_df['not_selected_test_acc'], eps, 1 - eps)
        res_df['not_selected_test_acc_probit'] = norm.ppf(res_df['not_selected_test_acc_clipped'])

        # Plot all three categories
        sns.scatterplot(data=res_df, x='train_acc_probit', y='selected_test_acc_probit', ax=axes[0], color=colors[0], label='Selected', alpha=0.7)
        sns.scatterplot(data=res_df, x='train_acc_probit', y='test_acc_probit', ax=axes[0], color=colors[1], label='All', alpha=0.7)
        sns.scatterplot(data=res_df, x='train_acc_probit', y='not_selected_test_acc_probit', ax=axes[0], color=colors[2], label='Not Selected', alpha=0.7)

        # Calculate regression lines for all three
        X_probit = res_df['train_acc_probit']
        Y_probit = res_df['test_acc_probit']
        slope, intercept, r_value, p_value, std_err = linregress(X_probit, Y_probit)

        selected_X_probit = res_df['train_acc_probit']
        selected_Y_probit = res_df['selected_test_acc_probit']
        selected_slope, selected_intercept, selected_r_value, selected_p_value, selected_std_err = linregress(selected_X_probit, selected_Y_probit)

        not_selected_X_probit = res_df['train_acc_probit']
        not_selected_Y_probit = res_df['not_selected_test_acc_probit']
        not_selected_slope, not_selected_intercept, not_selected_r_value, not_selected_p_value, not_selected_std_err = linregress(not_selected_X_probit, not_selected_Y_probit)

        # Set x-axis range to span the full range of all data points in probit space
        # Use the combined range of all three datasets
        all_x_values = np.concatenate([X_probit, selected_X_probit, not_selected_X_probit])
        x_min_probit = all_x_values.min()
        x_max_probit = all_x_values.max()
        x_range_probit = x_max_probit - x_min_probit
        x_min_extended = x_min_probit - 0.1 * x_range_probit  # Increased padding
        x_max_extended = x_max_probit + 0.1 * x_range_probit  # Increased padding

        # Plot regression lines
        xx = np.linspace(x_min_extended, x_max_extended, 200)
        yy = slope * xx + intercept
        axes[0].plot(xx, yy, linestyle='-', linewidth=2, color=colors[1], alpha=0.8)

        selected_xx = np.linspace(x_min_extended, x_max_extended, 200)
        selected_yy = selected_slope * selected_xx + selected_intercept
        axes[0].plot(selected_xx, selected_yy, linewidth=2, color=colors[0], alpha=0.8)

        not_selected_xx = np.linspace(x_min_extended, x_max_extended, 200)
        not_selected_yy = not_selected_slope * not_selected_xx + not_selected_intercept
        axes[0].plot(not_selected_xx, not_selected_yy, linewidth=2, color=colors[2], alpha=0.8)

        # Set the x-axis limits to ensure full range
        axes[0].set_xlim(x_min_extended, x_max_extended)

        annotation_text = (
            "Selected (All) [Not Selected]\n"
            f"Pearson R = {selected_r_value:.2f} ({r_value:.2f}) [{not_selected_r_value:.2f}]\n"
            f"Slope = {selected_slope:.2f} ({slope:.2f}) [{not_selected_slope:.2f}]\n"
            f"Intercept = {selected_intercept:.2f} ({intercept:.2f}) [{not_selected_intercept:.2f}]"
        )
        axes[0].text(
            0.05, 0.95,
            annotation_text,
            transform=axes[0].transAxes,
            va='top',
            fontsize=16,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        n_ticks = 6

        x_min = res_df['train_acc_clipped'].min()
        x_max = res_df['train_acc_clipped'].max()
        x_ticks_frac = np.linspace(x_min, x_max, n_ticks)
        x_ticks_probit = norm.ppf(x_ticks_frac)
        axes[0].set_xticks(x_ticks_probit)
        axes[0].set_xticklabels([f"{v:.2g}" for v in x_ticks_frac], rotation=90)

        y_min = res_df['test_acc_clipped'].min()
        y_max = res_df['test_acc_clipped'].max()
        y_ticks_frac = np.linspace(y_min, y_max, n_ticks)
        y_ticks_probit = norm.ppf(y_ticks_frac)
        axes[0].set_yticks(y_ticks_probit)
        axes[0].set_yticklabels([f"{v:.2g}" for v in y_ticks_frac])

        axes[0].tick_params(axis='x', labelsize=18)
        axes[0].tick_params(axis='y', labelsize=18)
        axes[0].set_xlabel("ID Acc", fontsize=24)
        axes[0].set_ylabel("OOD Acc", fontsize=24)
        axes[0].grid(True)
        axes[0].legend(fontsize=20, loc='lower left')

        # Increase figure size and add more padding
        fig.set_size_inches(10, 6)  # Larger figure
        fig.tight_layout(pad=3.0)  # More padding
        filepath = os.path.join(save_dir, f'{dataset}_model_accs_aotl_{train_idxs}_{test_env}_three_way_{num_ood}_{args.metric}.pdf')
        save_figure_if_needed(fig, filepath, args.force)
