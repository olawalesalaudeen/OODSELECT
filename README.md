# OODSelect

This repository contains the codebase for the paper:

**Aggregation Hides Out-of-Distribution Generalization Failures from Spurious Correlations** (NeurIPS 2025 Spotlight)

Olawale Salaudeen, Haoran Zhang, Kumail Alhamoud, Sara Beery, Marzyeh Ghassemi

Massachusetts Institute of Technology

Paper: [https://openreview.net/pdf?id=w97lDmoD0U](https://openreview.net/pdf?id=w97lDmoD0U)

---

## OODSelect Examples

The `OODSelect_examples/` folder contains pre-computed OOD sample selections at different sizes for various datasets. These selections were generated using the OODSelect method described in the paper and can be used directly for evaluation or analysis without running the full selection pipeline.

Each CSV file follows the naming convention `selected_files_{dataset}_{train_idxs}_{test_idx}_{num_samples}.csv` and contains the file paths or indices of selected OOD samples. The folder includes selections for datasets such as CXR_No_Finding, TerraIncognita, PACS, VLCS, WILDSCamelyon, and WILDSCivilComments, with selection sizes ranging from 10 to over 100,000 samples depending on the dataset. These examples demonstrate the OODSelect method's ability to identify well-specified OOD sets that maximize correlation between in-distribution accuracy and out-of-distribution performance.

---

## Acknowledgments

This codebase was formatted and improved for readability and usability with the assistance of a Cursor coding agent.

---

## Overview

This repository contains code for selecting well-specified OOD sets. This has a few applications:
1. Identifying well-specified OOD sets for evaluation domain generalization algorithms.
2. Identifying spurious correlations that harm performance.
3. Identifying samples in data that are most impacted by spurious correlations.

This repository contains code for few steps:
1. We need to train models; we can launch a job to train many models.
2. We need to take those train models and use their performance on ID/(maybe)OOD examples to select an OOD set.
3. We need to generate natural language descriptions of the difference between the ID and selected OOD set.

## Workflow and Data Flow

The repository follows a three-stage pipeline:

### Stage 1: Training ID Models
**Scripts**:
- `scripts/train_ID_models.py` - Train a single model
- `scripts/launch_train_ID_models.py` - Launch batch training jobs
**Inputs**:
- Dataset (e.g., TerraIncognita, PACS, WILDSCamelyon)
- Model architecture and training hyperparameters
- Training environment indices

**Outputs**:
- Trained model checkpoints (saved in `--output_dir`)
- Prediction CSV files containing:
  - Model predictions on validation/test sets
  - Per-sample accuracy flags
  - Class probabilities for each sample
- Wandb/TensorBoard logs (if logging enabled)
- A `done` file indicating job completion

**Output Format**: Each trained model produces a CSV file with columns:
- `domain`: Environment identifier (e.g., `env0_out`, `env3_in`)
- `label`: Ground truth label
- `p_0`, `p_1`, ...: Class probabilities for each class
- Additional metadata columns

### Stage 2: OOD Selection
**Scripts**:
- `scripts/OODSelect.py` - Run a single OOD selection job
- `scripts/launch_OODSelect.py` - Launch batch OOD selection jobs
- `scripts/generate_correlation_plots.py` - Generate correlation plots
- `scripts/generate_scatter_plots.py` - Generate scatter plots
- `scripts/launch_figures_jobs.py` - Launch batch figure generation jobs
**Inputs**:
- `--results_dir`: Directory containing prediction CSVs from Stage 1
- Training and test environment indices
- Number of OOD samples to select
- Loss type (Pearson R or R²)

**Outputs**:
- `min_acc_selection_{loss_type}.pt`: PyTorch tensor containing the selected OOD sample indices
- Wandb logs with:
  - Best trial hyperparameters
  - Validation and test correlations
  - Training progress
- A `done` file indicating job completion

**How it works**:
- Loads prediction CSVs from multiple trained models
- Extracts binary correctness flags for OOD samples (X) and ID accuracies (y)
- Uses Optuna to optimize sample selection weights that maximize correlation
- Returns the top-k samples with highest selection scores

### Stage 3: Natural Language Description
**Scripts**:
- `scripts/compare_ID_OOD_samples.py` - Compare ID and OOD samples using natural language
**Inputs**:
- `--selection_path`: Path to the selection tensor from Stage 2 (`min_acc_selection_{loss_type}.pt`)
- Dataset and environment indices
- Number of ID and OOD samples to compare

**Outputs**:
- Difference captions: List of natural language descriptions highlighting differences between ID and OOD sets
- Similarity deltas: DataFrame ranking difference captions by their discriminative power
- Captions for ID and OOD samples (optional, for debugging)

**How it works**:
- Samples ID and OOD sets based on the selection vector
- Generates captions for both sets using BLIP2
- Uses an LLM to propose difference captions
- Uses CLIP to rank difference captions by how well they distinguish ID vs OOD

### Data Flow Diagram

```
Stage 1: Training
┌─────────────────┐
│ Raw Dataset     │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ scripts/train_ID_    │──► Model Checkpoints
│      models.py       │──► Prediction CSVs (*.csv)
└──────────────────────┘

Stage 2: OOD Selection
┌─────────────────┐
│ Prediction CSVs │
│  (from Stage 1) │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│ scripts/OODSelect.py │──► Selection Vector (min_acc_selection_*.pt)
└──────────────────────┘

Stage 3: Natural Language
┌─────────────────┐
│ Selection Vector│
│  (from Stage 2) │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│ scripts/compare_ID_OOD_samples  │──► Difference Captions
│              .py                 │──► Similarity Rankings
└──────────────────────────────────┘
```

## 1. Training ID Models Scripts
Generate models.

### 1.1. `train_ID_models.py`

The `train_ID_models.py` script allows you to train a single model with specific parameters:

```bash
python scripts/train_ID_models.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --dataset TerraIncognita \
    --algorithm ERM \
    --model_arch resnet50 \
    --test_envs 0 \
    --seed 0 \
    --steps 5000 \
    --checkpoint_freq 100 \
    --holdout_fraction 0.2 \
    --log_backend wandb
```

**Outputs**:
- Prediction CSV files (e.g., `predictions_epoch_*.csv`) containing model predictions and class probabilities
- Model checkpoints and training logs in `--output_dir`
- A `done` file indicating successful completion

Key arguments:
- `--data_dir`: Directory containing the dataset
- `--output_dir`: Directory to save model outputs (used as input `--results_dir` in Stage 2)
- `--dataset`: Dataset name (e.g., TerraIncognita, PACS, WILDSCamelyon)
- `--algorithm`: Training algorithm (e.g., ERM)
- `--model_arch`: Model architecture (e.g., resnet50)
- `--test_envs`: Environment indices to use for testing
- `--log_backend`: Logging backend (wandb, tensorboard, csv, or none)

### 1.2. Batch Training with `launch_train_ID_models.py`

The `launch_train_ID_models.py` script allows you to launch multiple training jobs with different configurations; these experiments are in `src/domainbed_experiments.py`:

```bash
python scripts/launch_train_ID_models.py launch \
    --experiment PACS_ERM_Transfer \
    --command_launcher local \
    --log_backend wandb
```

Key arguments:
- `command`: Action to perform (launch, delete_incomplete, delete_all)
- `--experiment`: Experiment configuration to use
- `--command_launcher`: How to launch jobs (local or slurm)
- `--log_backend`: Logging backend for all jobs

## 2. OODSelect Scripts
Select OOD sets

### 2.1. `OODSelect.py`

The `OODSelect.py` script allows you to evaluate models to select OOD sets; we also generate figures summarizing the selections:

```bash
python scripts/OODSelect.py \
    --dataset TerraIncognita \
    --results_dir /path/to/results \
    --loss_type r \
    --num_epochs 3000 \
    --num_trials 5 \
    --train_idxs 0-1-2 \
    --test_idx 3 \
    --output_dir ./results \
    --wandb_project test_spurious_correlations_data_selection
```

**Inputs**:
- `--results_dir`: Directory containing prediction CSV files from Stage 1 training

**Outputs**:
- `min_acc_selection_{loss_type}.pt`: PyTorch tensor with selected OOD sample indices (used as `--selection_path` in Stage 3)
- Wandb logs with optimization metrics and correlations
- A `done` file indicating successful completion

**Key arguments**:
- `--results_dir`: Path to directory containing prediction CSVs from trained models
- `--loss_type`: Loss function type (`r` for Pearson R or `r2` for Pearson R²)
- `--num_OOD_samples`: Number of OOD samples to select
- `--train_idxs`: Training environment indices (used to compute ID accuracy)
- `--test_idx`: Test environment index (OOD samples come from this environment)

### 2.2. Batch OOD Selection with `launch_OODSelect.py`

The `launch_OODSelect.py` script allows you to launch multiple OOD selection jobs with different configurations; these experiments are in `src/OODSelect_experiments.py`:

```bash
python scripts/launch_OODSelect.py launch \
    --experiment TerraIncognita \
    --command_launcher local
```

Key arguments:
- `command`: Action to perform (launch, delete_incomplete, delete_all)
- `--experiment`: Experiment configuration to use (from `src/OODSelect_experiments.py`)
- `--command_launcher`: How to launch jobs (local or slurm)

### 2.3. Generating Correlation Plots with `generate_correlation_plots.py`

The `generate_correlation_plots.py` script creates correlation plots showing how correlation changes with the number of OOD samples:

```bash
python scripts/generate_correlation_plots.py \
    --dataset TerraIncognita \
    --num_domains 4 \
    --train_idxs 0-1-2 \
    --test_idx 3 \
    --results_dir /path/to/results \
    --metric r \
    --training_ood_samples 200 \
    --min_samples 10 \
    --max_samples 800 \
    --output_dir ./figures \
    --explore_selection \
    --selection_threshold 0.5
```

**Outputs**:
- Correlation plots (PDF) showing correlation vs number of OOD samples
- Class frequency plots showing how class distribution changes with sample size
- JSON files with correlation data

### 2.4. Generating Scatter Plots with `generate_scatter_plots.py`

The `generate_scatter_plots.py` script creates scatter plots comparing train accuracy vs test accuracy for selected and all OOD samples:

```bash
python scripts/generate_scatter_plots.py \
    --dataset TerraIncognita \
    --train_idxs 0-1-2 \
    --test_idx 3 \
    --num_oods 250 500 1000 \
    --results_dir /path/to/results \
    --selection_base /path/to/selection/results \
    --save_dir ./scatter_plots \
    --metric r
```

**Outputs**:
- Scatter plots (PDF) showing train accuracy vs test accuracy
- Comparisons between selected OOD samples and all samples
- Correlation values displayed on plots

### 2.5. Batch Figure Generation with `launch_figures_jobs.py`

The `launch_figures_jobs.py` script allows you to launch batch jobs to generate correlation and scatter plots for multiple datasets and configurations:

```bash
python scripts/launch_figures_jobs.py \
    --create_figure_jobs \
    --create_scatter_plot_jobs \
    --metric r \
    --split_by_architecture
```

Key arguments:
- `--create_figure_jobs`: Create jobs for correlation plots
- `--create_scatter_plot_jobs`: Create jobs for scatter plots
- `--metric`: Correlation metric to use (`r` or `r2`)
- `--split_by_architecture`: Split results by model architecture
- `--force`: Force regeneration of existing plots

This script automatically creates SLURM jobs (or local jobs) for generating figures across all configured datasets and experiment settings.

## 3. Compare ID and OOD Samples with Natural Language

### 3.1. Comparing ID and OOD Samples with `compare_ID_OOD_samples.py`

The `compare_ID_OOD_samples.py` script generates natural language descriptions of differences between ID and OOD sets:

```bash
python scripts/compare_ID_OOD_samples.py \
    --dataset TerraIncognita \
    --train_envs 0 1 2 \
    --test_envs 3 \
    --selection_path /path/to/selection/vector \
    --num_ID_samples 200 \
    --num_OOD_samples 200 \
    --num_difference_captions 25 \
    --label_idx 0
```

**Inputs**:
- `--selection_path`: Path to the selection tensor from Stage 2 (`min_acc_selection_{loss_type}.pt`)
- Dataset and environment indices

**Outputs**:
- Difference captions: Natural language descriptions of properties that distinguish OOD from ID samples
- Similarity deltas DataFrame: Rankings of difference captions by discriminative power (saved as CSV)
- Individual captions for ID and OOD samples (optional)

**Key arguments**:
- `--dataset`: Dataset name
- `--train_envs`: List of environment indices used for training (ID)
- `--test_envs`: List of environment indices used for testing (OOD)
- `--selection_path`: Path to the selection vector file from Stage 2
- `--num_ID_samples`: Number of in-distribution samples to analyze
- `--num_OOD_samples`: Number of out-of-distribution samples to analyze
- `--num_difference_captions`: Number of difference captions to generate
- `--label_idx`: Specific label to analyze (optional) -- when not specified, labels are ignored for analysis

**Process**:
1. Loads the selection vector and creates splits between ID and OOD samples
2. Generates captions for both sets using BLIP2
3. Uses an LLM to generate difference captions highlighting key distinctions
4. Uses CLIP to compute similarity deltas and rank difference captions

## 4. Experiment Configurations

The repository includes predefined experiment configurations in two files:

### 4.1. Training Configurations (`domainbed_experiments.py`)

These configurations are for training ID models:

1. `WILDSCamelyon_ERM_Transfer`: Training on WILDSCamelyon dataset with ERM algorithm (transfer learning)
2. `WILDSCamelyon_ERM_Finetune`: Training on WILDSCamelyon dataset with ERM algorithm (full fine-tuning)
3. `WILDSFMoW_ERM_Transfer`: Training on WILDSFMoW dataset with ERM algorithm (transfer learning)
4. `WILDSFMoW_ERM_Finetune`: Training on WILDSFMoW dataset with ERM algorithm (full fine-tuning)
5. `PACS_ERM_Transfer`: Training on PACS dataset with ERM algorithm (transfer learning)
6. `PACS_ERM_Finetune`: Training on PACS dataset with ERM algorithm (full fine-tuning)
7. `VLCS_ERM_Transfer`: Training on VLCS dataset with ERM algorithm (transfer learning)
8. `VLCS_ERM_Finetune`: Training on VLCS dataset with ERM algorithm (full fine-tuning)
9. `CXR_No_Finding_ERM_Transfer`: Training on CXR_No_Finding dataset with ERM algorithm (transfer learning)
10. `CXR_No_Finding_ERM_Finetune`: Training on CXR_No_Finding dataset with ERM algorithm (full fine-tuning)
11. `TerraIncognita_ERM_Transfer`: Training on TerraIncognita dataset with ERM algorithm (transfer learning)
12. `TerraIncognita_ERM_Finetune`: Training on TerraIncognita dataset with ERM algorithm (full fine-tuning)
13. `WILDSCivilComments_ERM_Transfer`: Training on WILDSCivilComments dataset with ERM algorithm (transfer learning)

Each training configuration specifies:
- Dataset and algorithm settings
- Model architecture and pretrained weights
- Training parameters (trials, seeds, holdout fractions)
- Transfer learning vs full fine-tuning

### 4.2. OOD Selection Configurations (`OODSelect_experiments.py`)

These configurations are for running OOD selection experiments:

1. `CXR_No_Finding`: OOD selection on CXR_No_Finding dataset
2. `TerraIncognita`: OOD selection on TerraIncognita dataset
3. `PACS`: OOD selection on PACS dataset
4. `VLCS`: OOD selection on VLCS dataset
5. `WILDSCamelyon`: OOD selection on WILDSCamelyon dataset
6. `WILDSFMoW`: OOD selection on WILDSFMoW dataset

Each OOD selection configuration specifies:
- Dataset and results directory
- Training and test environment indices
- Number of epochs and trials
- Loss type (r or r2)
- Number of OOD samples to select

## 5. Configuration

Before running the code, you need to configure paths in `src/project_paths.py`:

```python
# Training results directory - contains prediction CSV files from trained models
RESULTS_DIR = ""  # Update this to your training results directory path

# OOD selection results directory - contains OOD selection outputs
SELECTION_DIR = ""  # Update this to your OOD selection results directory path

# Output directory for figures, tables, and other outputs
OUTPUT_DIR = ""  # Update this to your output directory path

# SLURM partition name (if using SLURM)
PARTITION_NAME = ""  # Update this to your SLURM partition name

# Python executable path
# Leave empty to use system 'python', or specify full path to conda/env Python
PYTHON_PATH = ""  # Example: "/path/to/conda/envs/your_env/bin/python" or leave empty for "python"
```

All paths are centralized in this file and used throughout the codebase. Update these values to match your environment before running any scripts.

## 6. Logging

The training process supports multiple logging backends:
- Weights & Biases (wandb)
- TensorBoard
- CSV
- None (console only)

## 7. Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Weights & Biases (optional)
- TensorBoard (optional)
- Other dependencies listed in requirements.txt

