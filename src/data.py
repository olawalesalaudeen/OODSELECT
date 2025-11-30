# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import os
import hashlib
import sys
from collections import OrderedDict
from numbers import Number
import operator

import numpy as np
import torch
from collections import Counter
from itertools import cycle
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Tuple, Dict

from utils import seed_hash
from datasets import DATASETS, get_dataset_class

def make_weights_for_balanced_classes(dataset):
    counts = Counter()
    classes = []
    for _, y in dataset:
        y = int(y)
        counts[y] += 1
        classes.append(y)

    n_classes = len(counts)

    weight_per_class = {}
    for y in counts:
        weight_per_class[y] = 1 / (counts[y] * n_classes)

    weights = torch.zeros(len(dataset))
    for i, y in enumerate(classes):
        weights[i] = weight_per_class[int(y)]

    return weights


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
         xi, yi = minibatches[i][0], minibatches[i][1]
         xj, yj = minibatches[j][0], minibatches[j][1]

         min_n = min(len(xi), len(xj))
         pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights,
                num_samples=batch_size,
                replacement=True
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        # We'll sample infinitely in this sampler
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True
        )

        # Use a persistent worker DataLoader but only if you really need it
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True,
            persistent_workers=True  # Only if you do step-based training repeatedly
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 2**31


class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        self.loader = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            shuffle=False,
            batch_size=batch_size,
            drop_last=False
        ))

        self._length = len(self.loader)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.loader)

    def __len__(self):
        return self._length


class CombinedDataLoader:
    """DataLoader that yields batches from multiple dataloaders simultaneously."""
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        # Reset iterators and exhausted state
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.exhausted = [False] * len(self.dataloaders)
        return self

    def __next__(self):
        # Get next batch from each dataloader, or None if exhausted
        batches = []
        for i, (it, is_exhausted) in enumerate(zip(self.iterators, self.exhausted)):
            if is_exhausted:
                batches.append(None)
            else:
                try:
                    batches.append(next(it))
                except StopIteration:
                    self.exhausted[i] = True
                    batches.append(None)

        # If all dataloaders are exhausted, stop iteration
        if all(self.exhausted):
            raise StopIteration

        return batches

    def __len__(self):
        """Return the maximum length of any dataloader."""
        return max(len(dl) for dl in self.dataloaders)

class NamedDataLoader:
    def __init__(self, dataloader, name):
        """
        :param dataloader: The underlying PyTorch DataLoader (infinite or finite).
        :param name: The environment/domain name.
        """
        self.dataloader = dataloader
        self.name = name

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class DataModule(LightningDataModule):
    def __init__(self, args, hparams):
        super().__init__()
        self.args = args
        self.hparams.update(hparams)
        self.dataset = None
        # We’ll store train / val splits
        self.in_splits = []
        self.out_splits = []

    def setup(self, stage: Optional[str] = None):
        if self.dataset is not None:
            return
        if self.args.dataset in DATASETS:
            dataset_class = get_dataset_class(self.args.dataset)
            self.dataset = dataset_class(
                self.args.data_dir,
                self.args.test_envs,
                self.hparams
            )
        else:
            raise NotImplementedError(f"Dataset {self.args.dataset} not implemented")

        # Create splits for each environment
        for env_i, env in enumerate(self.dataset):
            # Split into out (validation) and in (training) sets
            out, in_ = split_dataset(
                env,
                int(len(env) * self.args.holdout_fraction),
                seed_hash(self.args.trial_seed, env_i)
            )
            print('in', env_i,len(in_))
            print('out', env_i, len(out))

            # Create balanced class weights if needed
            if self.hparams.get('class_balanced', False):
                in_weights = make_weights_for_balanced_classes(in_)
            else:
                in_weights = None

            # Store splits
            self.in_splits.append((in_, in_weights))
            self.out_splits.append((out, None))

        print("Data module setup complete.")

    def train_dataloader(self):
        """
        Return a single CombinedDataLoader that yields one batch from each environment
        at every step, via infinite loaders.
        """
        loaders = []
        for i, (env_ds, env_weights) in enumerate(self.in_splits):
            if i in self.args.test_envs:
                # skip test env from training
                continue
            loader = InfiniteDataLoader(
                dataset=env_ds,
                weights=env_weights,
                batch_size=self.hparams['batch_size'],
                num_workers=min(self.dataset.N_WORKERS, os.cpu_count()),
            )
            loaders.append(NamedDataLoader(loader, name=f"env{i}_in"))

        # Combine them so each step yields [batch_env0, batch_env1, ...]
        return CombinedDataLoader(loaders)

    def val_dataloader(self):
        """
        Instead of combining them, we return a list of separate loaders
        for each environment's validation set. This helps avoid memory blowups.
        """
        loaders = []
        val_batch_size = self.hparams.get('val_batch_size', 128)
        if (self.args.dataset in ['PACS', 'TerraIncognita', 'VLCS']) or (self.args.dataset == 'WILDSFMoW' and (4 in self.args.test_envs or 5 in self.args.test_envs)):
            for i, (env_ds, _) in enumerate(self.in_splits):
                if i not in self.args.test_envs:
                    continue
                dl = DataLoader(
                    env_ds,
                    batch_size=val_batch_size,
                    shuffle=False,
                    num_workers=min(2, os.cpu_count() or 1),
                    pin_memory=False,
                    persistent_workers=True
                )
                loaders.append(NamedDataLoader(dl, name=f"env{i}_in"))

        for i, (env_ds, _) in enumerate(self.out_splits):
            dl = DataLoader(
                env_ds,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=min(self.dataset.N_WORKERS, os.cpu_count()),
                pin_memory=False,
                persistent_workers=True
            )
            loaders.append(NamedDataLoader(dl, name=f"env{i}_out"))

        # Lightning will handle them as a list, calling validation_step per loader
        return loaders

    @property
    def input_shape(self):
        """Return the shape of input data."""
        return self.dataset.input_shape  # type: ignore

    @property
    def num_classes(self):
        """Return the number of classes."""
        return self.dataset.num_classes  # type: ignore

    @property
    def num_domains(self):
        """Return the number of domains."""
        return len(self.dataset)  # type: ignore
