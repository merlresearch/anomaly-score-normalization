# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import abstractmethod

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics

from network import (
    AdaProj,
    FFT_branch,
    STFT_branch,
    STFT_extractor,
    categorical_cross_entropy,
    feature_exchange,
    mixup_data,
    shallow_classifier,
)
from utils import get_ASD_performance, get_mean_embs, get_scores


class base_emb_model(L.LightningModule):
    def __init__(self, K: int = None, r: float = None):
        super().__init__()
        self.use_mse = False
        self.K = K
        self.r = r
        self.save_hyperparameters()

        # tensors needed for evaluation
        self.train_embs = []
        self.train_machine_ids = []
        self.train_source_labels = []
        self.means = []
        self.mean_machine_ids = []
        self.mean_source_labels = []
        self.val_embs = []
        self.val_machine_ids = []
        self.val_source_labels = []
        self.val_anomaly_labels = []
        self.test_embs = []
        self.test_machine_ids = []
        self.test_source_labels = []
        self.test_anomaly_labels = []

    @abstractmethod
    def forward(self, x):
        pass

    def reload_training_embs(self, batch, batch_nb):
        x, y, machine_id, source = batch
        x = x.to("cuda")
        machine_id = machine_id.to("cuda")
        source = source.to("cuda")
        embs_orig = self.get_normalized_embs_(x)
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        return

    @abstractmethod
    def training_step(self, batch, batch_nb):
        pass

    @abstractmethod
    def get_normalized_embs_(self, x):
        pass

    def validation_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs = self.get_normalized_embs_(x)
        self.val_embs.append(embs.detach())
        self.val_machine_ids.append(machine_id.detach())
        self.val_source_labels.append(source.detach())
        self.val_anomaly_labels.append(y.detach())
        return

    def test_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs = self.get_normalized_embs_(x)
        self.test_embs.append(embs.detach())
        self.test_machine_ids.append(machine_id.detach())
        self.test_source_labels.append(source.detach())
        self.test_anomaly_labels.append(y.detach())
        return

    def predict_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs = self.get_normalized_embs_(x)
        source_estimate = None  # source --> change between using ground truth or not, doesn't make a big difference
        scores = get_scores(
            torch.cat(self.train_embs, dim=0),
            torch.cat(self.train_machine_ids, dim=0),
            torch.cat(self.train_source_labels, dim=0),
            embs,
            machine_id,
            source_estimate,
            use_mse=self.use_mse,
            K=self.K,
            r=self.r,
        )  # this is the old approach without k-means
        return scores, machine_id, source, y

    def on_validation_epoch_end(self):
        # apply k-means on all training samples
        self.means, self.mean_machine_ids, self.mean_source_labels = (
            get_mean_embs(
                self.train_embs,
                self.train_machine_ids,
                self.train_source_labels,
                k=self.subspace_dim,
            )
        )

        # compute performance
        aauc, apauc, amauc, hauc, hpauc, hmauc = get_ASD_performance(
            self.means,
            self.mean_machine_ids,
            self.mean_source_labels,
            self.val_embs,
            self.val_machine_ids,
            self.val_source_labels,
            self.val_anomaly_labels,
            print_results=True,
        )
        log_values = {
            "amean_val_AUC": aauc,
            "amean_val_pAUC": apauc,
            "amean_val_mAUC": amauc,
            "hmean_val_AUC": hauc,
            "hmean_val_pAUC": hpauc,
            "hmean_val_mAUC": hmauc,
        }
        self.log_dict(log_values, prog_bar=True, on_epoch=True, on_step=False)

        # free memory
        self.val_embs.clear()
        self.val_machine_ids.clear()
        self.val_source_labels.clear()
        self.val_anomaly_labels.clear()
        return

    def on_test_epoch_end(self):
        # compute performance
        aauc, apauc, amauc, hauc, hpauc, hmauc = get_ASD_performance(
            self.means,
            self.mean_machine_ids,
            self.mean_source_labels,
            self.test_embs,
            self.test_machine_ids,
            self.test_source_labels,
            self.test_anomaly_labels,
            print_results=True,
        )

        # free memory
        self.test_embs.clear()
        self.test_machine_ids.clear()
        self.test_source_labels.clear()
        self.test_anomaly_labels.clear()
        return

    def on_train_epoch_end(self):
        # free memory
        self.train_embs.clear()
        self.train_machine_ids.clear()
        self.train_source_labels.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-3, weight_decay=1e-5, eps=1e-7
        )  # adjust settings to keras default


class raw_system(base_emb_model):
    def __init__(self, K: int = None, r: float = None):
        super().__init__(K=K, r=r)
        self.use_mse = True

    def forward(self, x):
        return x

    def training_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        embs_orig = self(x)
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        return None

    def get_normalized_embs_(self, x):
        embs = self(x)  # no normalization for raw embeddings
        return embs


class trainedACTsystem(base_emb_model):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        subspace_dim: int = 16,
        bias: bool = True,
        affine: bool = True,
        trainable_centers: bool = False,
        K: int = None,
        r: float = None,
    ):
        super().__init__(K=K, r=r)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim
        self.bias = bias
        self.affine = affine
        self.trainable_centers = trainable_centers

        # network modules
        self.emb_net = shallow_classifier(
            bias=self.bias,
            affine=self.affine,
            emb_dim=128,
            input_dim=self.input_dim,
        )
        self.adaproj = AdaProj(
            emb_dim=128,
            num_classes=self.num_classes,
            subspace_dim=self.subspace_dim,
            trainable=self.trainable_centers,
        )

    def forward(self, x):
        return self.emb_net(x)

    def training_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        # data augmentation
        x_mix, y_mix = mixup_data(x, y, p=1)
        # compute loss and accuracy
        embs_orig = self(x)
        embs_mix = self(x_mix)
        logits = self.adaproj(embs_mix, y_mix)
        loss = categorical_cross_entropy(logits, y_mix)
        embs_orig = F.normalize(embs_orig, p=2.0, dim=1)  # normalize
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        log_values = {"loss": loss}
        self.log_dict(log_values, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def get_normalized_embs_(self, x):
        embs = self(x)
        embs = F.normalize(embs, p=2.0, dim=1)  # normalize
        return embs


class ASDsystem(base_emb_model):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        subspace_dim: int = 16,
        bias: bool = True,
        affine: bool = True,
        trainable_centers: bool = False,
        nfft: int = 1024,
        use_fft: bool = True,
        temporal_normalization: bool = True,
        K: int = None,
        r: float = None,
    ):
        super().__init__(K=K, r=r)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim
        self.bias = bias
        self.affine = affine
        self.trainable_centers = trainable_centers
        self.nfft = nfft
        self.use_fft = use_fft
        self.temporal_normalization = temporal_normalization
        self.stft = STFT_extractor(
            affine=self.affine,
            nfft=self.nfft,
            temporal_normalization=self.temporal_normalization,
        )
        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.num_classes
        )

        # network modules
        self.stft_branch = STFT_branch(bias=self.bias, affine=self.affine)
        if self.use_fft:
            self.fft_branch = FFT_branch(
                input_dim=self.input_dim, bias=self.bias, affine=self.affine
            )
            self.adaproj_featex = AdaProj(
                emb_dim=512,
                num_classes=3 * self.num_classes,
                subspace_dim=self.subspace_dim,
                trainable=True,
            )
            self.adaproj = AdaProj(
                emb_dim=512,
                num_classes=self.num_classes,
                subspace_dim=self.subspace_dim,
                trainable=self.trainable_centers,
            )
        else:
            self.adaproj = AdaProj(
                emb_dim=256,
                num_classes=self.num_classes,
                subspace_dim=self.subspace_dim,
                trainable=self.trainable_centers,
            )

    def forward(self, x):
        if len(x.size()) == 2:
            emb_stft = self.stft_branch(self.stft(x))
        else:
            emb_stft = self.stft_branch(x)
        if self.use_fft:
            emb_fft = self.fft_branch(x)
            return emb_fft, emb_stft
        else:
            return emb_stft

    def training_step(self, batch, batch_nb):
        x, y, machine_id, source = batch
        # data augmentation
        x_mix, y_mix = mixup_data(x, y, p=0.5)
        # compute loss and accuracy
        if self.use_fft:
            embs_fft, embs_stft = self(x)
            embs_orig = torch.cat((embs_fft, embs_stft), dim=1)
            embs_fft_mix, embs_stft_mix = self(x_mix)
            embs_mix = torch.cat((embs_fft_mix, embs_stft_mix), dim=1)
            embs_featex, y_featex = feature_exchange(
                embs_fft_mix, embs_stft_mix, y_mix, p=0.5
            )
            logits_featex = self.adaproj_featex(embs_featex, y_featex)
            logits = self.adaproj(embs_mix, y_mix)
            loss = categorical_cross_entropy(
                logits, y_mix
            ) + categorical_cross_entropy(logits_featex, y_featex)
        else:
            embs_orig = self(x)
            embs_mix = self(x_mix)
            logits = self.adaproj(embs_mix, y_mix)
            loss = categorical_cross_entropy(logits, y_mix)
        embs_orig = F.normalize(embs_orig, p=2.0, dim=1)  # normalize
        self.train_machine_ids.append(machine_id.detach())
        self.train_embs.append(embs_orig.detach())
        self.train_source_labels.append(source.detach())
        log_values = {"loss": loss}
        self.log_dict(log_values, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def get_normalized_embs_(self, x):
        if self.use_fft:
            embs_fft, embs_stft = self(x)
            embs = torch.cat((embs_fft, embs_stft), dim=1)
        else:
            embs = self(x)
        embs = F.normalize(embs, p=2.0, dim=1)  # normalize
        return embs
