import sys
import hashlib
import torch
import operator
import json
import tqdm
import numpy as np
from collections import OrderedDict
from numbers import Number
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import os
import copy
import shlex
import command_launchers
import socket
from project_paths import get_python_path

class Job:
    """Class to manage individual training jobs."""
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args,
                 command_launcher='local'):
        keys = ['dataset', 'algorithm', 'test_envs', 'model_arch', 'transfer', 'weights', 'trial_seed', 'hparams_seed']
        python_path = get_python_path()
        args_str = json.dumps({k: train_args[k] for k in keys}, sort_keys=True)
        args_hash = f"{train_args['dataset']}_{train_args['algorithm']}_" + hashlib.md5(args_str.encode()).hexdigest()

        self.output_dir = os.path.join(
            train_args['output_dir'], args_hash
        )
        train_args['output_dir'] = self.output_dir

        self.train_args = copy.deepcopy(train_args)
        self.command_launcher = command_launcher
        command = ["cd .;",]
        command += [python_path, 'train_ID_models.py']

        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            elif isinstance(v, bool):
                v = ''
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    @staticmethod
    def launch(jobs, launcher_fn, max_slurm_jobs=1):
        """Launch the job if it hasn't been launched yet."""
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        commands = [job.command_str for job in jobs]
        output_dirs = [job.output_dir for job in jobs]
        if launcher_fn == command_launchers.slurm_launcher:
            launcher_fn(commands, output_dirs, max_slurm_jobs)
        else:
            launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['weights'],
            self.train_args['model_arch'],
            'transfer' if self.train_args['transfer'] else 'no_transfer',
            self.train_args['test_envs'],
            self.train_args['hparams_seed'],
            self.train_args['trial_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    def is_done(self):
        """Check if the job is complete."""
        return os.path.exists(os.path.join(self.output_dir, 'done'))

    def mark_done(self):
        """Mark the job as complete."""
        with open(os.path.join(self.output_dir, 'done'), 'w') as f:
            f.write('')
        self.state = Job.DONE


def seed_hash(*args):
    """Generate a seed from a hash of the arguments."""
    s = str(args)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**31)


def accuracy(network, loader, weights, device):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset : weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

def permute_top_k(
    x: torch.Tensor,
    k: int,
    d: int,
    seed: int = 0,
):
    """
    Args:
        x:      1-D tensor of real values, shape (n,).
        k:      how many top entries to permute.
        d:      how many permuted copies to produce.
        seed:   an integer seed for reproducibility (optional).

    Returns:
        A list of `d` tensors, each of shape (n,).  In each output tensor,
        the top-k entries of x are shuffled among themselves; all other
        entries remain exactly as in x.
    """
    gen = torch.Generator().manual_seed(seed)

    # find the indices of the top-k values
    topk = torch.topk(x, k)
    idx = topk.indices               # shape (k,)
    vals = x[idx]                    # the top-k values

    out = []
    for _ in range(d):
        # get a random permutation of [0..k-1]
        perm = torch.randperm(k, generator=gen) if gen is not None else torch.randperm(k)
        shuffled = vals[perm]
        y = x.clone()
        y[idx] = shuffled
        out.append(y)
    return out
