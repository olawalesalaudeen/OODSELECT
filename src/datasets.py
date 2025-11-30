# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import numpy as np
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset, DataLoader
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from pytorch_lightning import LightningDataModule
from typing import Optional, List, Tuple, Dict
from collections import Counter
from transformers import BertTokenizer, AutoTokenizer, DistilBertTokenizer, GPT2Tokenizer

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset

from utils import seed_hash
import string
from pathlib import Path
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # Spawrious datasets
    "SpawriousO2O_easy",
    "SpawriousO2O_medium",
    "SpawriousO2O_hard",
    "SpawriousM2M_easy",
    "SpawriousM2M_medium",
    "SpawriousM2M_hard",
    # other WILDS
    "WILDSAmazon",
    "WILDSCivilComments",
    'MNLI',
    'CXR'
]

CXR_TASKS = [
    'No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema'
]

for i in CXR_TASKS:
    DATASETS.append(f"CXR_{i.replace(' ', '_')}")

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        if hparams['weights'] == 'IMAGENET1K_V1':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_FEATURES':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_SWAG_E2E_V1':
            h, w = 384, 384
        elif hparams['weights'] == 'IMAGENET1K_SWAG_LINEAR_V1':
            h, w = 224, 224
        elif hparams['weights'] == 'random':
            h, w = 224, 224
        else:
            raise ValueError(f"Invalid weights: {hparams['weights']}")

        transform = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(h, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["C", "L", "S", "V"]
    NUM_CLASSES = 5
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "S"]
    NUM_CLASSES = 7
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    NUM_CLASSES = 10
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 100
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class XrayDataset():
    def __init__(self, root, dataset_name, target_name, test, target_shape = 224):
        self.dir = os.path.join(root, dataset_name)
        self.df = pd.read_csv(os.path.join(self.dir, 'spurious_selection_meta', 'metadata.csv'))
        self.target_name = target_name
        self.dataset_name = dataset_name

        if test:
            self.transform_ = transforms.Compose([
                transforms.Resize(target_shape),
                transforms.CenterCrop(target_shape),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_ = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(target_shape, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def transform(self, x):
        if self.dir.startswith(''):
            x = x.replace('', '')

        if 'MIMIC' in self.dataset_name:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-5] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            if reduced_img_path.is_file():
                x = str(reduced_img_path.resolve())
        elif self.dataset_name in ['vindr-cxr']:
            reduced_img_path = list(Path(x).parts)
            reduced_img_path[-2] = 'downsampled_files'
            reduced_img_path = Path(*reduced_img_path).with_suffix('.png')
            assert reduced_img_path.is_file()
            x = str(reduced_img_path.resolve())

        if self.dataset_name in ['PadChest']:
            img = np.array(Image.open(x))
            img = np.uint8(img/(2**16)*255)
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
            return self.transform_(Image.fromarray(img))
        else:
            return self.transform_(Image.open(x).convert("RGB"))

    def __getitem__(self, index):
        row = self.df.iloc[index]
        x = self.transform(row['filename'])
        y = torch.tensor(row[self.target_name], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.df)


class CXR(MultipleDomainDataset):
    ENVIRONMENTS = ["MIMIC-CXR-JPG", "CheXpert-v1.0-small", "ChestXray8", "PadChest", "vindr-cxr"]
    CHECKPOINT_FREQ = 200
    NUM_CLASSES = 2

    def __init__(self, root, test_envs, hparams, task):
        self.datasets = []

        if hparams['weights'] == 'IMAGENET1K_V1':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_FEATURES':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_SWAG_E2E_V1':
            h, w = 384, 384
        elif hparams['weights'] == 'IMAGENET1K_SWAG_LINEAR_V1':
            h, w = 224, 224
        elif hparams['weights'] == 'random':
            h, w = 224, 224
        else:
            raise ValueError(f"Invalid weights: {hparams['weights']}")

        self.input_shape = (3, h, w)
        self.num_classes = 2

        for c, env in enumerate(self.ENVIRONMENTS):
            if c in test_envs:
                self.datasets.append(XrayDataset(root, env, task, True, h))
            else:
                self.datasets.append(XrayDataset(root, env, task, False, h))
        super().__init__()

    def __getitem__(self, index):
        return self.datasets[index]

class CXR_No_Finding(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'No Finding')

class CXR_Atelectasis(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Atelectasis')

class CXR_Cardiomegaly(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Cardiomegaly')

class CXR_Effusion(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Effusion')

class CXR_Pneumonia(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Pneumonia')

class CXR_Pneumothorax(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Pneumothorax')

class CXR_Consolidation(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Consolidation')

class CXR_Edema(CXR):
    def __init__(self, root, test_envs, hparams):
        super().__init__(root, test_envs, hparams, 'Edema')

class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        if hparams['weights'] == 'IMAGENET1K_V1':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_V2':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_FEATURES':
            h, w = 224, 224
        elif hparams['weights'] == 'IMAGENET1K_SWAG_E2E_V1':
            h, w = 384, 384
        elif hparams['weights'] == 'IMAGENET1K_SWAG_LINEAR_V1':
            h, w = 224, 224
        elif hparams['weights'] == 'random':
            h, w = 224, 224
        else:
            raise ValueError(f"Invalid weights: {hparams['weights']}")

        transform = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        augment_transform = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.RandomResizedCrop(h, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSTextDataset(WILDSDataset):
    def __init__(self, dataset, metadata_name, test_envs, hparams):
        super(MultipleDomainDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(hparams['model_arch'])
        if hparams['model_arch'] == 'gpt2':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = hparams.get('max_length', 512)
        self.input_shape = (self.max_length,)
        self.datasets = []

        for i, metadata_value in enumerate(self.metadata_values(dataset, metadata_name)):
            env_dataset = WILDSTextEnvironment(
                dataset,
                metadata_name,
                metadata_value,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            self.datasets.append(env_dataset)

        self.num_classes = dataset.n_classes


class WILDSTextEnvironment:
    def __init__(self, wilds_dataset, metadata_name, metadata_value, tokenizer, max_length):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, i):
        text = self.dataset.get_input(self.indices[i])
        y = self.dataset.y_array[self.indices[i]]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        for k in encoding.keys():
            encoding[k] = encoding[k].squeeze(0)

        return encoding, y

    def __len__(self):
        return len(self.indices)


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    NUM_CLASSES = 2
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    NUM_CLASSES = 62
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)

class WILDSAmazon(WILDSTextDataset):
    ENVIRONMENTS = list(range(10, 21, 1))
    N_WORKERS = 1

    def __init__(self, root, test_envs, hparams):
        dataset = AmazonDataset(root_dir=root)

        year_idx = dataset.metadata_fields.index("year")
        valid_years = torch.tensor(self.ENVIRONMENTS)
        mask = torch.isin(dataset.metadata_array[:, year_idx], valid_years)

        dataset._input_array = [x for i, x in enumerate(dataset._input_array) if mask[i]]
        dataset._y_array = dataset._y_array[mask]
        dataset._metadata_array = dataset._metadata_array[mask]
        dataset._split_array = dataset._split_array[mask]

        super().__init__(
            dataset=dataset,
            metadata_name="year",
            test_envs=test_envs,
            hparams=hparams
        )


class WILDSCivilComments(WILDSTextDataset):
    ENVIRONMENTS = [
        'male',
        'female',
        'LGBTQ',
        'christian',
        'muslim',
        'other_religions',
        'black',
        'white'
    ]
    N_WORKERS = 1

    def __init__(self, root, test_envs, hparams):
        FLIP_RATIO = 0.2
        torch.manual_seed(42)
        dataset = CivilCommentsDataset(root_dir=root)

        identity_fields = self.ENVIRONMENTS
        n_samples = len(dataset.metadata_array)

        # Initialize new split array with -1
        new_split = torch.full((n_samples,), -1, dtype=torch.long)

        # For each sample, collect all environments where it has attribute=1
        for sample_idx in range(n_samples):
            possible_envs = []
            for env_idx, identity in enumerate(identity_fields):
                identity_idx = dataset.metadata_fields.index(identity)
                if dataset.metadata_array[sample_idx, identity_idx] == 1:
                    possible_envs.append(env_idx)

            # If sample has any matching environments, randomly assign to one of them
            if possible_envs:
                new_split[sample_idx] = possible_envs[torch.randint(len(possible_envs), (1,))]
            else:
                # If no matching environments, assign to random environment
                new_split[sample_idx] = torch.randint(len(identity_fields), (1,))

        # Randomly select 20% of samples to flip
        n_to_flip = int(FLIP_RATIO * n_samples)
        flip_indices = torch.randperm(n_samples)[:n_to_flip]

        # For each selected sample, assign to a random different environment
        for idx in flip_indices:
            current_env = new_split[idx]
            # Get list of other environments
            other_envs = [i for i in range(len(identity_fields)) if i != current_env]
            # Randomly select one
            new_env = other_envs[torch.randint(len(other_envs), (1,))]
            new_split[idx] = new_env

        # Add new_split to metadata
        dataset.metadata_fields.append('new_split')
        dataset._metadata_array = torch.cat([
            dataset._metadata_array,
            new_split.unsqueeze(1)
        ], dim=1)

        super().__init__(
            dataset=dataset,
            metadata_name="new_split",
            test_envs=test_envs,
            hparams=hparams
        )

class TextDataset():
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, i):
        text = self.df.iloc[i]['text']
        y = self.df.iloc[i]['label']

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        for k in encoding.keys():
            encoding[k] = encoding[k].squeeze(0)

        return encoding, int(y)

    def __len__(self):
        return len(self.df)

class MNLI(MultipleDomainDataset):
    ENVIRONMENTS = ['no_negation', 'negation']
    N_WORKERS = 1

    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.max_length = hparams.get('max_length', 512)
        self.input_shape = (self.max_length,)
        self.num_classes = 3

        root = Path(root)
        df = pd.concat((
            pd.read_json(root / 'MNLI/multinli_1.0_train.jsonl', lines = True),
            pd.read_json(root / 'MNLI/multinli_1.0_dev_matched.jsonl', lines = True),
            pd.read_json(root / 'MNLI/multinli_1.0_dev_mismatched.jsonl', lines = True)
        ))

        label_dict = {
            'contradiction': 0,
            'entailment': 1,
            'neutral': 2
        }

        df['label'] = df['gold_label'].map(label_dict)

        # adapted from https://github.com/kohpangwei/group_DRO/blob/master/dataset_scripts/generate_multinli.py
        negation_words = ['nobody', 'no', 'never', 'nothing']

        df['sentence2_has_negation'] = [False] * len(df)

        for negation_word in negation_words:
            df['sentence2_has_negation'] |= [negation_word in self.tokenize(sentence) for sentence in df['sentence2']]

        df['sentence2_has_negation'] = df['sentence2_has_negation'].astype(int)

        if hparams['model_arch'] == 'bert-base-uncased':
            # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['model_arch'])
        elif hparams['model_arch'] in ['xlm-roberta-base', 'allenai/scibert_scivocab_uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['model_arch'])
        elif hparams['model_arch'] == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif hparams['model_arch'] == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError

        if 'roberta' in hparams['model_arch']:
            self.CLS_TOKEN = '<s>'
            self.SEP_TOKEN = '</s>'
        elif 'bert' in hparams['model_arch']:
            self.CLS_TOKEN = '[CLS]'
            self.SEP_TOKEN = '[SEP]'
        else:
            self.CLS_TOKEN = 'Premise:'
            self.SEP_TOKEN = 'Hypothesis:'

        df['text'] = self.CLS_TOKEN + ' ' + df['sentence1'] + ' ' + self.SEP_TOKEN + ' ' + df['sentence2']

        df_neg = df[df['sentence2_has_negation'] == 1].sample(frac=1, random_state=42)
        df_nonneg = df[df['sentence2_has_negation'] == 0].sample(frac=1, random_state=42)

        # Indices at which to split
        neg_split = int(0.8 * len(df_neg))
        nonneg_split = int(0.2 * len(df_neg))

        # Environment 0 = 80% negation + 20% nonnegation
        df_env0 = pd.concat([
            df_neg.iloc[:neg_split],
            df_nonneg.iloc[:nonneg_split]
        ], ignore_index=True)

        # Environment 1 = leftover
        df_env1 = pd.concat([
            df_neg.iloc[neg_split:],
            df_nonneg.iloc[nonneg_split:]
        ], ignore_index=True)

        self.datasets = [TextDataset(df_env0, self.tokenizer, self.max_length),
                        TextDataset(df_env1, self.tokenizer, self.max_length)]

    def tokenize(self, s):
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.lower()
        s = s.split(' ')
        return s


## Spawrious base classes
class CustomImageFolder(Dataset):
    """
    A class that takes one folder at a time and loads a set number of images in a folder and assigns them a specific class
    """
    def __init__(self, folder_path, class_index, limit=None, transform=None):
        self.folder_path = folder_path
        self.class_index = class_index
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.class_index, dtype=torch.long)
        return img, label

class SpawriousBenchmark(MultipleDomainDataset):
    ENVIRONMENTS = ["Test", "SC_group_1", "SC_group_2"]
    input_shape = (3, 224, 224)
    num_classes = 4
    class_list = ["bulldog", "corgi", "dachshund", "labrador"]

    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, type1=False):
        self.type1 = type1
        train_datasets, test_datasets = self._prepare_data_lists(train_combinations, test_combinations, root_dir, augment)
        self.datasets = [ConcatDataset(test_datasets)] + train_datasets

    # Prepares the train and test data lists by applying the necessary transformations.
    def _prepare_data_lists(self, train_combinations, test_combinations, root_dir, augment):
        test_transforms = transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if augment:
            train_transforms = transforms.Compose([
                transforms.Resize((self.input_shape[1], self.input_shape[2])),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            train_transforms = test_transforms

        train_data_list = self._create_data_list(train_combinations, root_dir, train_transforms)
        test_data_list = self._create_data_list(test_combinations, root_dir, test_transforms)

        return train_data_list, test_data_list

    # Creates a list of datasets based on the given combinations and transformations.
    def _create_data_list(self, combinations, root_dir, transforms):
        data_list = []
        if isinstance(combinations, dict):

            # Build class groups for a given set of combinations, root directory, and transformations.
            for_each_class_group = []
            cg_index = 0
            for classes, comb_list in combinations.items():
                for_each_class_group.append([])
                for ind, location_limit in enumerate(comb_list):
                    if isinstance(location_limit, tuple):
                        location, limit = location_limit
                    else:
                        location, limit = location_limit, None
                    cg_data_list = []
                    for cls in classes:
                        path = os.path.join(root_dir, f"{0 if not self.type1 else ind}/{location}/{cls}")
                        data = CustomImageFolder(folder_path=path, class_index=self.class_list.index(cls), limit=limit, transform=transforms)
                        cg_data_list.append(data)

                    for_each_class_group[cg_index].append(ConcatDataset(cg_data_list))
                cg_index += 1

            for group in range(len(for_each_class_group[0])):
                data_list.append(
                    ConcatDataset(
                        [for_each_class_group[k][group] for k in range(len(for_each_class_group))]
                    )
                )
        else:
            for location in combinations:
                path = os.path.join(root_dir, f"{0}/{location}/")
                data = ImageFolder(root=path, transform=transforms)
                data_list.append(data)

        return data_list


    # Buils combination dictionary for o2o datasets
    def build_type1_combination(self,group,test,filler):
        total = 3168
        counts = [int(0.97*total),int(0.87*total)]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[0],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[1],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[2],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[3],counts[1])],
            ## filler
            ("bulldog","dachshund","labrador","corgi"):[(filler,total-counts[0]),(filler,total-counts[1])],
        }
        ## TEST
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[0]],
            ("dachshund",):[test[1], test[1]],
            ("labrador",):[test[2], test[2]],
            ("corgi",):[test[3], test[3]],
        }
        return combinations

    # Buils combination dictionary for m2m datasets
    def build_type2_combination(self,group,test):
        total = 3168
        counts = [total,total]
        combinations = {}
        combinations['train_combinations'] = {
            ## correlated class
            ("bulldog",):[(group[0],counts[0]),(group[1],counts[1])],
            ("dachshund",):[(group[1],counts[0]),(group[0],counts[1])],
            ("labrador",):[(group[2],counts[0]),(group[3],counts[1])],
            ("corgi",):[(group[3],counts[0]),(group[2],counts[1])],
        }
        combinations['test_combinations'] = {
            ("bulldog",):[test[0], test[1]],
            ("dachshund",):[test[1], test[0]],
            ("labrador",):[test[2], test[3]],
            ("corgi",):[test[3], test[2]],
        }
        return combinations

## Spawrious classes for each Spawrious dataset
class SpawriousO2O_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ["desert","jungle","dirt","snow"]
        test = ["dirt","snow","desert","jungle"]
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['mountain', 'beach', 'dirt', 'jungle']
        test = ['jungle', 'dirt', 'beach', 'snow']
        filler = "desert"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousO2O_hard(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['jungle', 'mountain', 'snow', 'desert']
        test = ['mountain', 'snow', 'desert', 'jungle']
        filler = "beach"
        combinations = self.build_type1_combination(group,test,filler)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'], type1=True)

class SpawriousM2M_easy(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['desert', 'mountain', 'dirt', 'jungle']
        test = ['dirt', 'jungle', 'mountain', 'desert']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])

class SpawriousM2M_medium(SpawriousBenchmark):
    def __init__(self, root_dir, test_envs, hparams):
        group = ['beach', 'snow', 'mountain', 'desert']
        test = ['desert', 'mountain', 'beach', 'snow']
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])

class SpawriousM2M_hard(SpawriousBenchmark):
    ENVIRONMENTS = ["Test","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        group = ["dirt","jungle","snow","beach"]
        test = ["snow","beach","dirt","jungle"]
        combinations = self.build_type2_combination(group,test)
        super().__init__(combinations['train_combinations'], combinations['test_combinations'], root_dir, hparams['data_augmentation'])
