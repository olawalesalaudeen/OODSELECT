import optuna
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer
from scipy.stats import linregress
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import pandas as pd
from itertools import combinations
from tqdm.auto import tqdm
import yaml
import glob
import pickle
import gc
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import wandb
import shlex
from copy import deepcopy
import command_launchers
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from project_paths import get_python_path as get_python_path_from_config

def probit_transform(x, eps=1e-6):
    """
    Compute the probit transform (inverse CDF of the standard normal) for tensor x.
    x should contain probabilities in the range (0, 1).
    """
    x = torch.clamp(x, eps, 1-eps)
    return math.sqrt(2) * torch.erfinv(2 * x - 1)

class PearsonR2Loss(nn.Module):
    """PyTorch module for computing Pearson R² loss."""

    def __init__(self, transform=probit_transform, eps=1e-6):
        super(PearsonR2Loss, self).__init__()
        self.transform = transform
        self.eps = eps

    def forward(self, X, y, z, eps=1e-3):
        """
        Compute Pearson R² loss.

        Args:
            X: Input feature matrix (shape: batch_size × features)
            y: Target vector (shape: batch_size)
            z: Feature weights (shape: features)

        Returns:
            The Pearson R² loss
        """
        z_sum = z.sum()
        x_z = (X @ z) / z_sum  # weighted features (shape: batch_size)
        x_z = torch.clip(x_z, min=eps, max=1-eps)

        # Apply transform before computing correlation
        x_t = self.transform(x_z)
        y_t = self.transform(y)

        # Center the data
        x_z_c = x_t - torch.mean(x_t)
        y_c = y_t - torch.mean(y_t)

        # Compute covariance
        cov = torch.sum(x_z_c * y_c) / x_z.shape[0]

        # Compute variances with epsilon for numerical stability
        x_var = torch.mean(x_z_c ** 2) + self.eps
        y_var = torch.mean(y_c ** 2) + self.eps

        # Compute correlation coefficient and return R²
        corr = cov / torch.sqrt(x_var * y_var)
        return corr ** 2


class PearsonRLoss(nn.Module):
    """PyTorch module for computing Pearson R loss."""

    def __init__(self, transform=probit_transform, eps=1e-6):
        super(PearsonRLoss, self).__init__()
        self.transform = transform
        self.eps = eps

    def forward(self, X, y, z, eps=1e-6):
        """
        Compute Pearson R loss.

        Args:
            X: Input feature matrix (shape: batch_size × features)
            y: Target vector (shape: batch_size)
            z: Feature weights (shape: features)

        Returns:
            The Pearson R loss
        """
        z_sum = z.sum() + 1e-6
        x_z = (X @ z) / z_sum  # weighted samples (shape: batch_size)

        # Apply transform before computing correlation
        x_t = self.transform(x_z)
        y_t = self.transform(y)

        # Center the data
        x_z_c = x_t - torch.mean(x_t)
        y_c = y_t - torch.mean(y_t)

        # Compute covariance
        cov = torch.mean(x_z_c * y_c)

        # Compute variances with epsilon for numerical stability
        x_var = torch.mean(x_z_c ** 2) + self.eps
        y_var = torch.mean(y_c ** 2) + self.eps

        # Compute and return correlation coefficient
        corr = cov / torch.sqrt(x_var * y_var)
        if torch.isnan(corr):
            raise ValueError("Correlation is NaN")
        return corr


class TestSetFinder(pl.LightningModule):
    """
    PyTorch Lightning module for optimizing feature weights to maximize correlation.

    This model learns a set of weights for OOD samples to maximize the Pearson correlation
    between a weighted combination of OOD samples and a target variable.
    """

    def __init__(self, num_samples, num_OOD_samples, loss_type='r',
                 transform=probit_transform, penalty=0.,
                 penalty_anneal_iters=1000, lr=0.01,
                 weight_decay=0.01, eta_min=0.01, T_max=100,
                 output_dir='./results/'):
        """
        Initialize the model.

        Args:
            num_samples: Number of samples to select
            num_OOD_samples: Number of OOD samples to select
            loss_type: Either 'r' for Pearson R loss or 'r2' for Pearson R² loss
            transform: Optional transform to apply to data before computing correlation
            lr: Learning rate for optimization
        """
        super(TestSetFinder, self).__init__()
        self.output_dir = output_dir
        # self.save_hyperparameters()

        # Initialize sample weights as model parameters
        self.selection = nn.Parameter(torch.randn(num_samples))

        # Choose loss function based on loss_type
        if loss_type.lower() == 'r2':
            self.loss_fn = PearsonR2Loss(transform=transform)
            self.c = 1
        else:
            self.loss_fn = PearsonRLoss(transform=transform)
            self.c = 1

        self.penalty = penalty
        self.penalty_anneal_iters = penalty_anneal_iters
        self.weight_decay = weight_decay
        self.eta_min = eta_min
        self.lr = lr
        self.T_max = T_max
        self.num_OOD_samples = num_OOD_samples
        self.loss_type = loss_type

    def forward(self, X, y):
        """Compute sample weights."""
        return self.loss_fn(X, y, torch.sigmoid(self.selection))

    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        X, y = batch

        # Compute correlation
        corr = self(X, y)
        # We clamp self.global_step so it doesn't exceed penalty_anneal_iters
        progress = min(float(self.global_step) / float(self.penalty_anneal_iters), 1.0)

        progress = self.global_step / float(self.penalty_anneal_iters)

        progress = min(max(progress, 0.0), 1.0)                  # clamp to [0,1]
        warmup_ratio = 0.5 * (1.0 - math.cos(math.pi * progress))  # 0→1
        current_penalty = self.penalty * warmup_ratio

        loss = self.c*corr + current_penalty * nn.functional.softplus(
            self.num_OOD_samples - torch.sigmoid(self.selection).sum()
        )

        # Log metrics
        self.log('train_correlation', corr)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning."""
        X, y = batch

        selection = TestSetFinder.get_binary_selection(self.selection, self.num_OOD_samples)
        # selection = self.selection
        # Compute correlation on validation data
        corr = self.loss_fn(X, y, selection).item()
        loss = corr

        # Log metrics
        self.log('val_correlation', corr)
        self.log('val_loss', loss)

        # return loss

    @staticmethod
    def test(test_set, train_accs, soft_selection, num_OOD_samples, loss_type,
             transform=probit_transform):
        if loss_type == 'r':
            loss_fn = PearsonRLoss(transform=transform)
        else:
            loss_fn = PearsonR2Loss(transform=transform)
        selection = TestSetFinder.get_binary_selection(soft_selection, num_OOD_samples)
        return loss_fn(test_set, train_accs, selection).item()

    # def configure_optimizers(self):
    #     """Configure optimizer."""
    #     return torch.optim.Adam([self.selection], lr=self.lr)
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam([self.selection], lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.Adam([self.selection], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.T_max, eta_min=self.eta_min)
        return [optimizer], [scheduler]

    def get_feature_importance(self):
        """Get normalized feature weights as importance scores."""
        weights = self.selection.detach()
        return weights / weights.sum()

    @staticmethod
    def get_binary_selection(selection, num_OOD_samples):
        """
        Return a binary mask where the top-k entries in `self.selection` are 1, others are 0.
        """
        k = num_OOD_samples
        selection = selection.detach()

        # Get indices of top-k values
        topk_indices = torch.topk(selection, k).indices

        # Create a binary mask
        binary_mask = torch.zeros_like(selection)
        binary_mask[topk_indices] = 1.0

        return binary_mask

    @staticmethod
    def tune_hparams(trial, train_loader, val_loader, num_epochs,
                     num_samples, num_OOD_samples, loss_type,
                     transform=probit_transform, output_dir='./results/'):
        lr = trial.suggest_loguniform('lr', 1e-2, 1e2)
        # weight_decay = trial.suggest_loguniform('weight_decay', 1e-3, 1e-1)
        eta_min = trial.suggest_loguniform('eta_min', 1e-3, 1e0)
        T_max = trial.suggest_int('T_max', 10, 200)
        penalty = trial.suggest_uniform('penalty', 1e-1, 1e2)
        penalty_anneal_iters = trial.suggest_int('penalty_anneal_iters', 10, 1000)

        model = TestSetFinder(
            num_samples=num_samples,
            num_OOD_samples=num_OOD_samples,
            loss_type=loss_type,
            penalty=penalty,
            penalty_anneal_iters=penalty_anneal_iters,
            lr=lr,
            # weight_decay=weight_decay,
            eta_min=eta_min,
            T_max=T_max,
            transform=transform,
            output_dir=output_dir
        )

        early_stop_callback = EarlyStopping(monitor='val_correlation',
                                            patience=500, mode='min')
        model_checkpoint = ModelCheckpoint(monitor='val_correlation',
                                            mode='min',
                                            filename='best_model',
                                            save_top_k=1,
                                            save_last=True,
                                            save_weights_only=True,
                                            )
        trainer = Trainer(
            max_epochs=num_epochs,
            callbacks=[early_stop_callback, model_checkpoint],
            log_every_n_steps=10,
            enable_progress_bar=True,
            default_root_dir=output_dir
        )

        trainer.fit(model, train_loader, val_loader)

        # Save the model's state dictionary in the trial attributes
        trial.set_user_attr("soft_selection", model.selection.detach().cpu().numpy())

        return trainer.callback_metrics['val_correlation'].item()


def plot_aotl(x, y):
    # Clip to avoid log/probit issues
    x = np.clip(x, 1e-6, None)
    y = np.clip(y, 1e-6, None)

    # Fit regression on original data
    res = linregress(x, y)
    # print(res)
    slope, intercept, r_value, p_value, std_err = res

    # Normalize and probit-transform for plotting positions
    def to_probit_scale(v):
        eps = 1e-6
        v_scaled = (v - v.min()) / (v.max() - v.min())
        v_scaled = np.clip(v_scaled, eps, 1 - eps)
        return norm.ppf(v_scaled)

    x_probit = to_probit_scale(x)
    y_probit = to_probit_scale(y)

    # Plot probit-transformed positions
    plt.figure(figsize=(6, 4))
    sns.regplot(x=x_probit, y=y_probit, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'}, label=f'{r_value:.2f}')

    # Set custom tick positions (in probit) and labels (in original)
    xticks_raw = np.linspace(x.min(), x.max(), 6)
    yticks_raw = np.linspace(y.min(), y.max(), 6)

    xticks_probit = to_probit_scale(xticks_raw)
    yticks_probit = to_probit_scale(yticks_raw)

    plt.xticks(xticks_probit, [f"{val:.2f}" for val in xticks_raw], rotation=270)
    plt.yticks(yticks_probit, [f"{val:.2f}" for val in yticks_raw], rotation=0)

    return res

def one_hot_accuracy(y_pred, y_true, num_classes=None):
    """
    Computes accuracy using one-hot encoding (NumPy version).

    Args:
        y_pred (ndarray): Predicted class labels (N,), integers.
        y_true (ndarray): True class labels (N,), integers.
        num_classes (int, optional): Total number of classes. If None, inferred.

    Returns:
        float: Accuracy value between 0 and 1.
    """
    if num_classes is None:
        num_classes = max(y_pred.max(), y_true.max()) + 1

    y_pred_one_hot = np.eye(num_classes)[y_pred]
    y_true_one_hot = np.eye(num_classes)[y_true]
    correct = np.sum(y_pred_one_hot * y_true_one_hot, axis=1)
    accuracy = np.mean(correct)
    return accuracy

# ----------------------------------------------------------------------
# Helper ­– runs in a worker process
# ----------------------------------------------------------------------
def _load_one_csv(args):
    """
    Read a single predictions CSV and return:
      • train‑environment accuracy
      • a list of (env_name, sampled_df) tuples for every requested test env
      • the list of probability‑column names (p_*)
    """
    (csv_path, train_envs, out_test_envs,
     max_total_samples, seed) = args

    df = pd.read_csv(csv_path)

    # discover columns that hold per‑class probabilities
    acc_cols = [c for c in df.columns if c.startswith("p_")]
    if not acc_cols:
        raise ValueError(f"No columns starting with 'p_' in {csv_path}")

    # concatenate all requested train environments
    train_df = pd.concat(
        [df[df.domain == f"env{env}_out"] for env in sorted(train_envs)],
        ignore_index=True
    )
    if train_df.empty:
        # nothing to score ⇒ skip this CSV
        return None

    train_acc = accuracy_score(
        train_df.label.values,
        train_df.loc[:, acc_cols].values.argmax(axis=1)
    )

    # gather & sample the requested test environments
    env_dfs = []
    for env in out_test_envs:
        sub = df[(df.domain == f"env{env}_in") | (df.domain == f"env{env}_out")]
        if sub.empty:
            continue
        n_sample = len(sub) if max_total_samples is None else min(len(sub), max_total_samples)
        sub = sub.sample(
            n_sample, random_state=seed
        )
        env_dfs.append((env, sub))

    # keep memory footprint per worker small
    del df
    gc.collect()

    return train_acc, env_dfs, acc_cols


# ----------------------------------------------------------------------
# Public API ­– parallel CSV loader
# ----------------------------------------------------------------------
def load_prediction_data(
    dataset: str,
    num_domains: int,
    results_dir: str,
    train_idxs: str,
    test_idx: str,
    max_total_samples = None,
    max_total_models = None,
    seed: int = 0,
):
    """
    Load model‑prediction CSVs in parallel and return:
        • out_data  : list[(train_acc, df)] – each df is *min‑truncated* so all equal length
        • acc_cols  : list[str]             – names of p_* probability columns
    """

    out_test_envs = (
        [str(i) for i in test_idx.split("-")]
        if "-" in str(test_idx)
        else [str(test_idx)]
    )
    train_envs = [str(i) for i in train_idxs.split("-")]
    test_envs  = [str(i) for i in range(num_domains) if str(i) not in train_envs]

    # ------------------------------------------------------------------ collect CSV paths
    csv_paths = []
    exp_dirs = os.listdir(results_dir)
    with tqdm(exp_dirs, desc=f"collecting {dataset} CSVs", unit="exp") as bar:
        for exp_dir in bar:
            if dataset.lower() not in exp_dir.lower():
                continue
            predictions_dir = Path(results_dir) / exp_dir

            cfg_path = predictions_dir / "wandb/latest-run/files/config.yaml"
            try:
                with cfg_path.open() as f:
                    cfg = yaml.safe_load(f)
                if not all(i in map(str, cfg["test_envs"]["value"])
                           for i in test_envs):
                    continue
            except FileNotFoundError:
                continue

            csvs = sorted(
                glob.glob(str(predictions_dir / "*.csv")),
                key=lambda p: int(Path(p).stem.split("_")[2]),
            )
            csv_paths.extend(csvs)

    rng = np.random.RandomState(seed)
    rng.shuffle(csv_paths)

    # ------------------------------------------------------------------ parse CSVs in parallel
    out_data = []
    acc_cols_global = None
    num_models = 0

    with ProcessPoolExecutor() as pool, \
         tqdm(total=len(csv_paths), desc=f"parsing {dataset} CSVs", unit="csv") as pbar:

        futures = {
            pool.submit(
                _load_one_csv,
                (p, train_envs, out_test_envs, max_total_samples, seed),
            ): p
            for p in csv_paths
        }

        for fut in as_completed(futures):
            pbar.update()              # one CSV finished
            result = fut.result()
            if result is None:
                continue

            train_acc, env_dfs, acc_cols_worker = result

            if acc_cols_global is None:
                acc_cols_global = acc_cols_worker
            elif acc_cols_worker != acc_cols_global:
                raise ValueError(
                    f"Mismatched p_* columns between CSV files."
                )

            for _, df in env_dfs:
                out_data.append((train_acc, df))
                num_models += 1

    if not out_data:
        raise ValueError(
            f"No data found for {dataset} with train_idxs={train_idxs} "
            f"and test_idx={test_idx}"
        )

    # ------------------------------------------------------------------ equal‑length truncation
    max_n = max(df.shape[0] for _, df in out_data)
    out_data = [(acc, df) for acc, df in out_data if df.shape[0] == max_n]

    return out_data, acc_cols_global

def extract_data(data, acc_cols):
    """
    Extracts a list of (training accuracy, OOD prediction DataFrame) tuples
    from datasets_lists using key '0' and subkey '0'.

    Inputs:
        data (list): List of tuples (preictions_df, training accuracy).
        acc_cols (list): List of columns starting with 'p_' (assumed identical across CSVs).

    Returns:
        X (numpy array): Matrix of shape (num_models, num_OOD_samples) where each row is a binary
            correctness flag for each OOD sample
        y (numpy array): Array of shape (num_models,) where each entry is the ID accuracy
    """
    out_data = []

    for ID_acc, OOD_predictions in tqdm(data, desc="Extracting accuracies and predictions"):
        OOD_correct_flag = (OOD_predictions.label.values ==
                            OOD_predictions.loc[:, acc_cols].values.argmax(axis=1)).astype(int)
        out_data.append((OOD_correct_flag, ID_acc))

    X = np.concatenate([x[0].reshape(1, -1) for x in out_data], axis=0)  # shape: (num_models, num_OOD_samples)
    y = np.array([x[1] for x in out_data])  # shape: (num_models,)

    return X, y


def prepare_dataset(X, y):
    """
    Prepares the data for training by splitting into train/val/test sets and converting to PyTorch DataLoaders.

    Args:
        X: numpy array of shape (num_models, num_OOD_samples) where each row is a binary
            correctness flag for each OOD sample
        y: numpy array of shape (num_models,) where each entry is the ID accuracy

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each set
        X_train_tensor, y_train_tensor: Training set tensors
        X_val_tensor, y_val_tensor: Validation set tensors
        X_test_tensor, y_test_tensor: Test set tensors for final evaluation
    """
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print('Train:', X_train.shape, y_train.shape)
    print('Val:', X_val.shape, y_val.shape)
    print('Test:', X_test.shape, y_test.shape)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Using full dataset batch size as in original code.
    batch_size = X_temp.shape[0]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor


def run_optuna(train_loader, val_loader, num_epochs, num_samples, num_OOD_samples, loss_type, n_trials=5, output_dir='./results'):
    """
    Run Optuna optimization with wandb logging.

    Inputs:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        num_epochs: Number of epochs to train for
        num_samples: Number of samples to select
        num_OOD_samples: Number of OOD samples to select
        loss_type: Type of loss function to use
    """
    study = optuna.create_study(direction='minimize')

    def objective(trial):
        # Log trial number
        wandb.log({"trial_number": trial.number})

        # Run hyperparameter tuning using existing method
        val_correlation = TestSetFinder.tune_hparams(
            trial, train_loader, val_loader, num_epochs,
            num_samples, num_OOD_samples, loss_type,
            transform=probit_transform,
            output_dir=output_dir
        )

        # Log trial results
        wandb.log({
            "val_correlation": val_correlation,
            "lr": trial.params.get('lr'),
            # "weight_decay": trial.params.get('weight_decay'),
            "eta_min": trial.params.get('eta_min'),
            "T_max": trial.params.get('T_max'),
            "penalty": trial.params.get('penalty'),
            "penalty_anneal_iters": trial.params.get('penalty_anneal_iters')
        })

        return val_correlation

    study.optimize(objective, n_trials=n_trials)

    # Log best trial details
    wandb.log({
        "best_trial_value": study.best_trial.value,
        "best_trial_params": study.best_trial.params,
        "best_trial_number": study.best_trial.number
    })

    # Save best model state
    best_model_state = study.best_trial.user_attrs["soft_selection"]
    model_state_path = os.path.join(output_dir, f"best_model_state_{loss_type}.pt")
    torch.save(best_model_state, model_state_path)

    return study


class TestJob:
    """Class to manage individual test jobs."""
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, test_args, python_path=None, command_launcher='local'):
        self.test_args = deepcopy(test_args)
        self.python_path = python_path
        self.command_launcher = command_launcher

        self.test_args['output_dir'] = os.path.join(self.test_args['output_dir'], f"Test_{self.test_args['dataset']}_{'-'.join(map(str, self.test_args['train_idxs']))}_{self.test_args['test_idx']}_{self.test_args['num_OOD_samples']}_{self.test_args['loss_type']}")

        command = ["cd .;"]
        # Use configured python path or 'python' as fallback if python_path is None or empty
        if not python_path:
            python_path = get_python_path_from_config()
        command += [python_path, 'OODSelect.py']

        for k, v in sorted(self.test_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            elif isinstance(v, bool):
                v = ''
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.test_args['output_dir'], 'done')):
            self.state = TestJob.DONE
        elif os.path.exists(self.test_args['output_dir']):
            self.state = TestJob.INCOMPLETE
        else:
            self.state = TestJob.NOT_LAUNCHED


    @staticmethod
    def launch(jobs, launcher_fn, max_slurm_jobs=1):
        """Launch the job if it hasn't been launched yet."""
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        commands = [job.command_str for job in jobs]
        output_dirs = [job.test_args['output_dir'] for job in jobs]
        if launcher_fn == command_launchers.slurm_launcher:
            launcher_fn(commands, output_dirs, max_slurm_jobs)
        else:
            launcher_fn(commands)

    def __str__(self):
        return f"TestJob(command_str={self.command_str}, state={self.state})"

    def is_done(self):
        return self.state == TestJob.DONE

    def mark_done(self):
        with open(os.path.join(self.test_args['output_dir'], 'done'), 'w') as f:
            f.write('')
        self.state = TestJob.DONE


