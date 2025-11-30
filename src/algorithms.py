import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
import pytorch_lightning as pl
import pandas as pd
import os

ALGORITHMS = [
    'ERM',
    'ERMPlusPlus',
    'Fish',
    'IRM',
    'GroupDRO',
    'Mixup',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'SD',
    'ANDMask',
    'SANDMask',
    'IGA',
    'SelfReg',
    "Fishr",
    'TRM',
    'IB_ERM',
    'IB_IRM',
    'CAD',
    'CondCAD',
    'Transfer',
    'CausIRL_CORAL',
    'CausIRL_MMD',
    'EQRM',
    'RDM',
    'ADRMX',
    'URM',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class AlgorithmModule(pl.LightningModule):
    def __init__(self, algorithm_class, input_shape, num_classes, num_domains, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.algorithm = algorithm_class(input_shape, num_classes, num_domains, hparams)

        self.predictions = []

    def training_step(self, batches, batch_idx):
        # batches is a list of (x, y) tuples, one from each dataloader
        step_vals = self.algorithm.update(batches, None)  # No UDA for now

        # Log training loss
        self.log("train/loss", step_vals['loss'], on_step=True, on_epoch=False, sync_dist=True)

        # Compute and log accuracy for each training environment
        env_accs = []
        for i, (x, y) in enumerate(batches):
            logits = self.algorithm.predict(x)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            env_accs.append(acc)
            # Get the name from the dataloader
            name = self.trainer.train_dataloader.dataloaders[i].name
            self.log(f"train/{name}/acc", acc, on_step=True, on_epoch=False, sync_dist=True)

        # Log average accuracy across all environments
        avg_env_acc = sum(env_accs) / len(env_accs)
        self.log("train/avg_acc", avg_env_acc, on_step=True, on_epoch=False, sync_dist=True)

        return step_vals['loss']  # Return the loss tensor directly


    def validation_step(self, batch, batch_idx, dataloader_idx):
        """
        Called once per batch, per validation dataloader.
        `dataloader_idx` tells us which loader we’re coming from.
        """
        x, y = batch

        with torch.no_grad():
            logits = self.algorithm.predict(x)
            preds = torch.argmax(logits, dim=1)

            name = self.trainer.val_dataloaders[dataloader_idx].name
            self.predictions.append((name, y.cpu().numpy().ravel(), logits.detach().cpu().numpy()))


    def on_validation_epoch_end(self):
        """
        `all_outputs` is a list of lists:
         - `len(all_outputs) = number_of_val_dataloaders`
         - `all_outputs[i]` is the list of outputs from dataloader i
        """
        env_accs = {}
        predictions = []
        for name, y, preds in self.predictions:
            if name is None or y is None or preds is None:
                continue
            # Accuracy
            if name not in env_accs:
                env_accs[name] = []

            # Predictions
            assert len(y) == len(preds)
            for y_i, pred_i in zip(y.tolist(), preds.tolist()):
                predictions.append([name, y_i] + pred_i)

        # Save predictions
        print(self.logger.save_dir)
        predictions_df = pd.DataFrame(predictions, columns=['domain', 'label'] + [f'p_{i}' for i in range(len(predictions[0])-2)])
        predictions_df.to_csv(os.path.join(self.logger.save_dir, f'global_step_{self.global_step}_predictions.csv'), index=False)

        # Compute and log average accuracy for each environment
        all_env_accs = []
        N = 0
        for env in predictions_df.domain.unique()  :
            accs = (predictions_df[predictions_df.domain == env].label.values == predictions_df[predictions_df.domain == env].iloc[:, 2:].values.argmax(axis=1)).astype(int)
            env_avg_acc = float(sum(accs)) / len(accs)
            # Get the name from the dataloader
            name = env
            self.log(f'val/{name}/acc', env_avg_acc, sync_dist=True)
            all_env_accs.append(env_avg_acc*len(accs))
            N += len(accs)

        overall_avg = sum(all_env_accs) / N
        self.log('val/avg_acc', overall_avg, sync_dist=True)
        self.log('val_acc', overall_avg, sync_dist=True)

        self.predictions.clear()

    def configure_optimizers(self):
        return self.algorithm.optimizer

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.hparams.update(hparams)

    def update(self, minibatches, unlabeled=None):
        if not torch.is_tensor(minibatches[0][0]):  # Text input
            # Concatenate all input_ids, attention_mask, etc.
            all_x = {
                k: torch.cat([x[k] for x, _ in minibatches])
                for k in minibatches[0][0].keys()
            }
        else:  # Image input
            all_x = torch.cat([x for x, _ in minibatches])

        all_y = torch.cat([y for _, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        return {'loss': loss}  # Return tensor instead of float

    def predict(self, x):
        return self.network(x)


