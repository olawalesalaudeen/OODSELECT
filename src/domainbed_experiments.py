import numpy as np
import logging
import copy
from itertools import product, combinations as itertools_combinations
from datasets import get_dataset_class
from model_weights_pairs import PRETRAINED_VARIANTS, TEXT_MODELS
from project_paths import get_output_dir

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
- WILDSCamelyon_ERM_Transfer: Transfer learning experiments
- WILDSFMoW_ERM_Transfer: Transfer learning experiments
- PACS_ERM_Transfer: Transfer learning experiments
- VLCS_ERM_Transfer: Transfer learning experiments
- TerraIncognita_ERM_Transfer: Transfer learning experiments
- WILDSCivilComments_ERM_Transfer: Transfer learning experiments
'''

# Constants
N_TRIALS = 1
N_HPARAMS = 1

class WILDSCamelyon_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('WILDSCamelyon').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3, 4]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['WILDSCamelyon'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [True],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        # Create weight variants for each combination
        weight_variants = []
        for combo in output_combinations:
            if combo['model_arch'] in PRETRAINED_VARIANTS:
                for weight in PRETRAINED_VARIANTS[combo['model_arch']]:
                    combo_copy = combo.copy()
                    combo_copy['weights'] = weight
                    weight_variants.append(combo_copy)
            else:
                weight_variants.append(combo)
        return weight_variants

class WILDSCamelyon_ERM_Finetune:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('WILDSCamelyon').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3, 4]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['WILDSCamelyon'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [False],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        # Create weight variants for each combination
        weight_variants = []
        for combo in output_combinations:
            if combo['model_arch'] in PRETRAINED_VARIANTS:
                for weight in PRETRAINED_VARIANTS[combo['model_arch']]:
                    combo_copy = combo.copy()
                    combo_copy['weights'] = weight
                    weight_variants.append(combo_copy)
            else:
                weight_variants.append(combo)
        return weight_variants

class WILDSFMoW_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('WILDSFMoW').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3, 4, 5]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['WILDSFMoW'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [True],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class WILDSFMoW_ERM_Finetune:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('WILDSFMoW').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3, 4, 5]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['WILDSFMoW'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [False],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class PACS_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('PACS').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['PACS'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [True],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class PACS_ERM_Finetune:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('PACS').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['PACS'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [False],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class VLCS_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('VLCS').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[1]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['VLCS'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [True],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class VLCS_ERM_Finetune:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('VLCS').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[1]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['VLCS'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [False],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class CXR_No_Finding_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('CXR_No_Finding').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[0]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['CXR_No_Finding'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [True],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class CXR_No_Finding_ERM_Finetune:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('CXR_No_Finding').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[0]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['CXR_No_Finding'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [False],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class TerraIncognita_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('TerraIncognita').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = []
        test_envs_list = [[3]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['TerraIncognita'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [True],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class TerraIncognita_ERM_Finetune:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('TerraIncognita').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[3]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['TerraIncognita'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': list(PRETRAINED_VARIANTS.keys()),
                'transfer': [False],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),  # 3 trials
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        updated_combinations = []
        for combination in output_combinations:
            model = combination['model_arch']
            if model in PRETRAINED_VARIANTS:
                variants = PRETRAINED_VARIANTS[model]
                for variant in variants:
                    new_combination = combination.copy()
                    new_combination['weights'] = variant
                    updated_combinations.append(new_combination)
        return updated_combinations

class WILDSCivilComments_ERM_Transfer:
    fname = 'train_ID_models.py'

    @staticmethod
    def hparams():
        num_domains = len(get_dataset_class('WILDSCivilComments').ENVIRONMENTS)
        # Generate all possible combinations of test environments
        test_envs_list = [[4,7]]

        grid = {
            'data': {
                'data_dir': [''],
                'output_dir': [get_output_dir()],
                'dataset': ['WILDSCivilComments'],
                'test_envs': [list(envs) for envs in test_envs_list],  # Convert tuples to lists
            },
            'model': {
                'algorithm': ['ERM'],
                'model_arch': TEXT_MODELS,
                'transfer': [True],
                'weights': [None],
            },
            'training': {
                'trial_seed': list(range(N_TRIALS)),
                'hparams_seed': list(range(N_HPARAMS)),
                'holdout_fraction': [0.2],
                'uda_holdout_fraction': [0.0],
            },
        }

        # Flatten the nested grid
        flat_grid = {}
        for section, params in grid.items():
            flat_grid.update(params)

        output_combinations = combinations(flat_grid)
        return output_combinations

