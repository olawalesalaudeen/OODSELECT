"""
Centralized configuration for all project paths.

Update these paths to match your environment before running the code.
"""

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


def get_results_dir():
    """Get the training results directory."""
    return RESULTS_DIR


def get_selection_dir():
    """Get the OOD selection results directory."""
    return SELECTION_DIR


def get_output_dir():
    """Get the output directory for figures and tables."""
    return OUTPUT_DIR


def get_partition_name():
    """Get the SLURM partition name."""
    return PARTITION_NAME


def get_python_path():
    """Get the Python executable path.

    Returns the configured PYTHON_PATH if set, otherwise returns 'python'.
    """
    return PYTHON_PATH if PYTHON_PATH else "python"

