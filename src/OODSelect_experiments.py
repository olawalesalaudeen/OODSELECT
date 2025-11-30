import numpy as np
import logging
from itertools import product, combinations as itertools_combinations
from datasets import get_dataset_class
from project_paths import get_output_dir, get_results_dir

logging.basicConfig(level='WARNING')

def combinations(grid):
    """Returns list of dictionaries for all possible combinations in grid."""
    return list(dict(zip(grid.keys(), values)) for values in product(*grid.values()))

def get_hparams(experiment):
    """Get hyperparameters for a specific experiment."""
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].hparams()

def get_script_name(experiment):
    """Get the script name for a specific experiment."""
    if experiment not in globals():
        raise NotImplementedError(experiment)
    return globals()[experiment].fname

#### write experiments here
'''
Experimental order:
- TerraIncognita
- PACS
- VLCS
- WILDSCamelyon
- WILDSFMoW
- CXR_No_Finding
'''

# Constants
N_TRIALS = 100
class CXR_No_Finding:
    fname = 'OODSelect.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],  # Update to your data directory
                'output_dir': [get_output_dir()],
                'python_path': [''],  # Leave empty to use get_python_path() from project_paths
                'dataset': ['CXR_No_Finding'],
                'results_dir': [get_results_dir()],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 100000, 250000],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[1,2,3,4]]:
                for test_idx in [0]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)
        return output_combinations


class TerraIncognita:
    fname = 'OODSelect.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],  # Update to your data directory
                'output_dir': [get_output_dir()],
                'python_path': [''],  # Leave empty to use get_python_path() from project_paths
                'dataset': ['TerraIncognita'],
                'results_dir': [get_results_dir()],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations


class PACS:
    fname = 'OODSelect.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('PACS').ENVIRONMENTS)
        grid = {
            'data': {
                'data_dir': [''],  # Update to your data directory
                'output_dir': [get_output_dir()],
                'python_path': [''],  # Leave empty to use get_python_path() from project_paths
                'dataset': ['PACS'],
                'results_dir': [get_results_dir()],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations

class VLCS:
    fname = 'OODSelect.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('VLCS').ENVIRONMENTS)
        grid = {
            'data': {
                'data_dir': [''],  # Update to your data directory
                'output_dir': [get_output_dir()],
                'python_path': [''],  # Leave empty to use get_python_path() from project_paths
                'dataset': ['VLCS'],
                'results_dir': [get_results_dir()],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,2,3]]:
                for test_idx in [1]:
                    if test_idx not in train_idxs:
                        combo_i = combo.copy()
                        combo_i['train_idxs'] = train_idxs
                        combo_i['test_idx'] = test_idx
                        output_combinations.append(combo_i)

        return output_combinations

class WILDSCamelyon:
    fname = 'OODSelect.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],  # Update to your data directory
                'output_dir': [get_output_dir()],
                'python_path': [''],  # Leave empty to use get_python_path() from project_paths
                'dataset': ['WILDSCamelyon'],
                'results_dir': [get_results_dir()],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 75000, 100000],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3, 4]:
                    if test_idx not in train_idxs:
                            combo_i = combo.copy()
                            combo_i['train_idxs'] = train_idxs
                            combo_i['test_idx'] = test_idx
                            output_combinations.append(combo_i)

        return output_combinations

class WILDSFMoW:
    fname = 'OODSelect.py'

    @staticmethod
    def hparams():
        grid = {
            'data': {
                'data_dir': [''],  # Update to your data directory
                'output_dir': [get_output_dir()],
                'python_path': [''],  # Leave empty to use get_python_path() from project_paths
                'dataset': ['WILDSFMoW'],
                'results_dir': [get_results_dir()],
            },
            'training': {
              'num_epochs': [50000],
              'num_trials': [N_TRIALS],
              'loss_type': ['r'],
              'num_OOD_samples': [10, 20, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 75000, 100000, 150000],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = []
        base_combinations = combinations(flat_grid)

        for combo in base_combinations:
            for train_idxs in [[0,1,2]]:
                for test_idx in [3]:
                    if test_idx not in train_idxs:
                        combo_i = combo.copy()
                        combo_i['train_idxs'] = train_idxs
                        combo_i['test_idx'] = test_idx
                        output_combinations.append(combo_i)

        return output_combinations

