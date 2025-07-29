# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def keras_initialization(layer):
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return


def mixup_data(
    x, y, alpha: float = 1.0, p: float = 1.0, use_cuda: bool = True
):
    if p < 0 or p > 1:
        raise ValueError(
            "mixup probability has to be between 0 and 1, but got {}".format(p)
        )

    # sample random parameters
    batch_size = x.size()[0]
    lam = torch.rand((batch_size, 1)).cuda()
    apply_idx = (torch.rand(batch_size) < p).cuda()

    # apply mixup
    mixed_x = x.flip(dims=[0])
    mixed_y = y.flip(dims=[0])
    mixed_x[apply_idx] = (lam * mixed_x.flip(dims=[0]) + (1 - lam) * mixed_x)[
        apply_idx
    ]
    mixed_y[apply_idx] = (lam * mixed_y.flip(dims=[0]) + (1 - lam) * mixed_y)[
        apply_idx
    ]
    return mixed_x, mixed_y


def feature_exchange(x1, x2, y, p):
    # randomly exchange representations of two branches before concatenating them
    if p < 0 or p > 1:
        raise ValueError(
            "exchange probability has to be between 0 and 1, but got {}".format(
                p
            )
        )

    # sample random parameters
    batch_size = x1.size()[0]
    apply_idx = (torch.rand(batch_size) < p).cuda()

    # apply feature exchange
    exchanged_x1 = x1.flip(dims=[0])
    exchanged_y = torch.cat(
        [y, torch.zeros_like(y), torch.zeros_like(y)], dim=1
    )
    exchanged_x1[~apply_idx] = x1[~apply_idx]
    exchanged_y[apply_idx] = torch.cat(
        [
            torch.zeros_like(y[apply_idx]),
            0.5 * y[apply_idx],
            0.5 * y.flip(dims=[0])[apply_idx],
        ],
        dim=1,
    )
    x_cat = torch.cat((exchanged_x1, x2), dim=1)
    return x_cat, exchanged_y


class Conv2dSame(nn.Conv2d):
    # https://github.com/pytorch/pytorch/issues/67551
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(
            i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0]
        )
        pad_w = self.calc_same_pad(
            i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                ],
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class FFT_branch(nn.Module):
    def __init__(self, input_dim: int, bias: bool = True, affine: bool = True):
        super().__init__()
        self.bias = bias
        self.affine = affine
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=128,
            kernel_size=256,
            stride=64,
            bias=self.bias,
        )
        # keras_initialization(self.conv1)
        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=64,
            stride=32,
            bias=self.bias,
        )
        # keras_initialization(self.conv2)
        self.conv3 = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=16,
            stride=4,
            bias=self.bias,
        )
        # keras_initialization(self.conv3)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(
            in_features=self._get_flatten_dimension(),
            out_features=128,
            bias=self.bias,
        )
        # keras_initialization(self.lin1)
        self.norm1 = nn.BatchNorm1d(128, affine=self.affine)
        self.lin2 = nn.Linear(
            in_features=128, out_features=128, bias=self.bias
        )
        # keras_initialization(self.lin2)
        self.norm2 = nn.BatchNorm1d(128, affine=self.affine)
        self.lin3 = nn.Linear(
            in_features=128, out_features=128, bias=self.bias
        )
        # keras_initialization(self.lin3)
        self.norm3 = nn.BatchNorm1d(128, affine=self.affine)
        self.lin4 = nn.Linear(
            in_features=128, out_features=128, bias=self.bias
        )
        # keras_initialization(self.lin4)
        self.norm4 = nn.BatchNorm1d(128, affine=self.affine)
        self.emb = nn.Linear(in_features=128, out_features=256, bias=self.bias)
        # keras_initialization(self.emb)

    def _get_flatten_dimension(self):
        dim = (
            self.input_dim / 2
        )  # assuming that only one half of the full spectrum will be used
        dim = self._get_output_dim_for_conv(dim, 256, 64, 0)
        dim = self._get_output_dim_for_conv(dim, 64, 32, 0)
        dim = self._get_output_dim_for_conv(dim, 16, 4, 0)
        dim = dim * 128  # multiply with number of channels
        return dim

    def _get_output_dim_for_conv(
        self, input_dim, filter_size, stride, pad_size
    ):
        return int(
            np.floor(input_dim + 2 * pad_size - filter_size) / stride + 1
        )

    def forward(self, x):
        # pre-process data
        x = torch.abs(torch.fft.fft(x)[:, : int(x.size()[-1] / 2)])
        x = torch.unsqueeze(
            x, dim=1
        )  # add channel dimension, which follows after batch dimension for PyTorch

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # dense layers
        x = self.flatten(x)
        x = F.relu(self.norm1(self.lin1(x)))
        x = F.relu(self.norm2(self.lin2(x)))
        x = F.relu(self.norm3(self.lin3(x)))
        x = F.relu(self.norm4(self.lin4(x)))
        emb = self.emb(x)  # embedding layer
        return emb


class shallow_classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        bias: bool = True,
        affine: bool = True,
        emb_dim: int = 128,
    ):
        super().__init__()
        self.bias = bias
        self.affine = affine
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.norm0 = nn.BatchNorm1d(self.input_dim, affine=self.affine)
        self.lin1 = nn.Linear(
            in_features=self.input_dim, out_features=512, bias=self.bias
        )
        self.norm1 = nn.BatchNorm1d(512, affine=self.affine)
        self.lin2 = nn.Linear(
            in_features=512, out_features=128, bias=self.bias
        )
        self.norm2 = nn.BatchNorm1d(128, affine=self.affine)
        self.drop = nn.Dropout(p=0.5)
        self.emb = nn.Linear(
            in_features=128, out_features=self.emb_dim, bias=self.bias
        )

    def forward(self, x):
        # dense layers
        x = self.norm0(x)
        x = F.relu(self.norm1(self.lin1(x)))
        x = F.relu(self.norm2(self.lin2(x)))
        emb = self.emb(self.drop(x))  # embedding layer
        return emb


class Residual_block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        affine: bool = True,
        input_norm: bool = True,
        use_strides: bool = True,
    ):
        super().__init__()
        self.bias = bias
        self.affine = affine
        self.input_norm = input_norm
        self.use_strides = use_strides
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.BatchNorm2d(self.in_channels, affine=self.affine)
        if self.use_strides:
            self.conv1 = Conv2dSame(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=2,
                kernel_size=3,
                bias=self.bias,
            )
        else:
            self.conv1 = Conv2dSame(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                bias=self.bias,
            )
        # keras_initialization(self.conv1)
        self.norm2 = nn.BatchNorm2d(self.out_channels, affine=self.affine)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="same",
            bias=self.bias,
        )
        # keras_initialization(self.conv2)
        if self.use_strides:
            self.pool = nn.MaxPool2d(1, 2)
            self.conv_down = Conv2dSame(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                padding="same",
                bias=self.bias,
            )
            # keras_initialization(self.conv_down)
        self.norm3 = nn.BatchNorm2d(self.out_channels, affine=self.affine)
        self.conv3 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="same",
            bias=self.bias,
        )
        # keras_initialization(self.conv3)
        self.norm4 = nn.BatchNorm2d(self.out_channels, affine=self.affine)
        self.conv4 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="same",
            bias=self.bias,
        )
        # keras_initialization(self.conv4)

    def forward(self, x):
        if self.input_norm:
            x = self.norm1(x)
        xr = self.conv2(F.relu(self.norm2(self.conv1(F.relu(x)))))
        if self.use_strides:
            x = self.norm3(self.conv_down(self.pool(x)) + xr)
        else:
            x = self.norm3(x + xr)
        xr = self.conv4(F.relu(self.norm4(self.conv3(F.relu(x)))))
        return x + xr


class STFT_extractor(nn.Module):
    def __init__(
        self,
        affine: bool = True,
        nfft: int = 1024,
        temporal_normalization=True,
    ):
        super().__init__()
        self.nfft = nfft
        self.affine = affine
        self.norm_inp = nn.BatchNorm1d(
            513, affine=self.affine
        )  # note that this is 1D batch normalization, i.e. temporal normalization
        self.temporal_normalization = temporal_normalization

    def forward(self, x):
        stft = torch.abs(
            torch.stft(
                x,
                n_fft=self.nfft,
                hop_length=512,
                window=torch.hann_window(
                    self.nfft,
                    device=torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    ),
                ),
                center=False,
                normalized=False,
                return_complex=True,
            )
        )  # dimension is 513x561 (freq x time)
        if self.temporal_normalization:
            stft = stft - torch.mean(
                stft, axis=2, keepdim=True
            )  # temporal mean normalization
        stft = self.norm_inp(
            stft
        )  # only used for variance, mean is zero for every sample because of the previous line
        return torch.transpose(stft, 1, 2)


class STFT_branch(nn.Module):
    def __init__(
        self, bias: bool = True, affine: bool = True, nfft: int = 1024
    ):
        super().__init__()
        self.nfft = nfft
        self.bias = bias
        self.affine = affine
        self.conv1 = Conv2dSame(
            in_channels=1,
            out_channels=16,
            kernel_size=7,
            stride=2,
            bias=self.bias,
        )
        # keras_initialization(self.conv1)
        self.norm1 = nn.BatchNorm2d(16, affine=self.affine)
        self.pool1 = nn.MaxPool2d(3, stride=2)
        self.res_block1 = Residual_block(
            in_channels=16,
            out_channels=16,
            bias=self.bias,
            affine=self.affine,
            input_norm=False,
            use_strides=False,
        )
        self.res_block2 = Residual_block(
            in_channels=16, out_channels=32, bias=self.bias, affine=self.affine
        )
        self.res_block3 = Residual_block(
            in_channels=32, out_channels=64, bias=self.bias, affine=self.affine
        )
        self.res_block4 = Residual_block(
            in_channels=64,
            out_channels=128,
            bias=self.bias,
            affine=self.affine,
        )
        self.flatten = nn.Flatten()
        self.norm2 = nn.BatchNorm1d(2048, affine=self.affine)
        self.emb = nn.Linear(
            in_features=2048, out_features=256, bias=self.bias
        )
        # keras_initialization(self.emb)

    def forward(self, x):
        # pre-process data
        x = torch.unsqueeze(
            x, dim=1
        )  # add channel dimension, which follows after batch dimension for PyTorch

        # convolutional layers
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # dense layers
        x = torch.max(x, dim=2).values  # max-pooling over time
        x = self.flatten(x)
        x = self.norm2(x)
        emb = self.emb(x)  # embedding layer
        return emb


def categorical_cross_entropy(logits, labels, from_logits=True):
    if from_logits:
        return -(F.log_softmax(logits, dim=1) * labels).sum(dim=1).mean()
    else:
        return -(torch.log(logits) * labels).sum(dim=1).mean()


class AdaProj(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        subspace_dim: int = 1,
        trainable: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim
        self.trainable = trainable
        self.eps = eps
        self.W = nn.Parameter(
            data=torch.Tensor(
                self.num_classes, self.subspace_dim, self.emb_dim
            ),
            requires_grad=self.trainable,
        )
        nn.init.xavier_uniform_(self.W)
        self.s = nn.Parameter(
            data=torch.Tensor(1), requires_grad=False
        )  # manual update step
        nn.init.constant_(
            self.s,
            np.maximum(np.sqrt(2.0) * np.log(self.num_classes - 1.0), 0.5),
        )
        self.pi = nn.Parameter(
            data=torch.Tensor(1), requires_grad=False
        )  # constant pi
        nn.init.constant_(self.pi, math.pi)

    def forward(self, x, labels=None):
        x = F.normalize(x, p=2.0, dim=1)  # batchsize x emb_dim
        W = F.normalize(
            self.W, p=2.0, dim=2
        )  # num_classes x subspace_dim x emb_dim
        logits = torch.tensordot(
            x, W, dims=[[1], [2]]
        )  # batchsize x num_classes x subspace_dim
        x_proj = (
            torch.unsqueeze(logits, dim=3) * torch.unsqueeze(W, dim=0)
        ).sum(
            dim=2
        )  # batchsize x num_classes x emb_dim
        x_proj = F.normalize(x_proj, p=2.0, dim=2)
        logits = (torch.unsqueeze(x, dim=1) * x_proj).sum(
            dim=2
        )  # batchsize x num_classes
        if labels is None:
            return logits
        if self.training:
            self.update_scale_parameter(logits, labels)
        return logits * self.s

    def update_scale_parameter(self, logits, labels):
        theta = torch.acos(
            torch.clamp(logits, min=-1.0 + self.eps, max=1.0 - self.eps)
        )
        max_logits = torch.max(self.s * logits)
        B_avg = torch.where(
            labels < 1,
            torch.exp(self.s * logits - max_logits),
            torch.exp(torch.zeros_like(self.s * logits) - max_logits),
        )
        B_avg = B_avg.sum(dim=1).mean()
        theta_class = torch.sum(
            labels * theta, dim=1
        )  # take mix-upped angle of mix-upped classes
        theta_med = torch.median(theta_class)
        self.s = nn.Parameter(
            torch.clamp(
                (max_logits + torch.log(B_avg))
                / torch.cos(torch.minimum(self.pi / 4, theta_med)),
                min=self.eps,
            ),
            requires_grad=False,
        )


class SCAdaCos(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        num_subclusters: int = 16,
        trainable: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.num_subclusters = num_subclusters
        self.trainable = trainable
        self.W = nn.Parameter(
            data=torch.Tensor(
                self.num_classes * self.num_subclusters, self.emb_dim
            ),
            requires_grad=self.trainable,
        )
        nn.init.xavier_uniform_(self.W)
        self.s = nn.Parameter(
            data=torch.Tensor(1), requires_grad=False
        )  # manual update step
        nn.init.constant_(
            self.s,
            np.maximum(
                np.sqrt(2.0)
                * np.log(self.num_classes * self.num_subclusters - 1.0),
                0.5,
            ),
        )
        self.pi = nn.Parameter(
            data=torch.Tensor(1), requires_grad=False
        )  # constant pi
        nn.init.constant_(self.pi, math.pi)

    def forward(self, x, labels=None):
        x = F.normalize(x, p=2.0, dim=1)  # batchsize x emb_dim
        W = F.normalize(self.W, p=2.0, dim=1)  # num_classes x emb_dim
        logits = torch.tensordot(
            x, W, dims=[[1], [1]]
        )  # batchsize x num_classes
        if labels is None:
            return logits
        with torch.no_grad():
            self.update_scale_parameter(logits, labels)
        return logits * self.s

    def update_scale_parameter(self, logits, labels, eps=1e-12):
        theta = torch.acos(torch.clamp(logits, min=-1.0 + eps, max=1.0 - eps))
        max_logits = torch.max(self.s * logits)
        labels = torch.repeat_interleave(
            labels, repeats=self.num_subclusters, dim=1
        )
        B_avg = torch.exp(
            self.s * logits - max_logits
        )  # diverges without using mixup
        B_avg = B_avg.sum(dim=1).mean()
        theta_class = torch.sum(
            labels * theta, dim=1
        )  # take mix-upped angle of mix-upped classes
        theta_med = torch.median(theta_class)
        self.s = nn.Parameter(
            torch.clamp(
                (max_logits + torch.log(B_avg))
                / torch.cos(torch.minimum(self.pi / 4, theta_med)),
                min=eps,
            ),
            requires_grad=False,
        )

    def compute_softmax_probs(self, logits):
        out = F.softmax(logits - torch.max(logits), dim=1)
        out = torch.reshape(out, (-1, self.num_classes, self.num_subclusters))
        return torch.sum(out, dim=2)


class AdaCos(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, trainable: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.trainable = trainable
        self.W = nn.Parameter(
            data=torch.Tensor(self.num_classes, self.emb_dim),
            requires_grad=self.trainable,
        )
        nn.init.xavier_uniform_(self.W)
        self.s = nn.Parameter(
            data=torch.Tensor(1), requires_grad=False
        )  # manual update step
        nn.init.constant_(
            self.s,
            np.maximum(np.sqrt(2.0) * np.log(self.num_classes - 1.0), 0.5),
        )
        self.pi = nn.Parameter(
            data=torch.Tensor(1), requires_grad=False
        )  # constant pi
        nn.init.constant_(self.pi, math.pi)

    def forward(self, x, labels=None):
        x = F.normalize(x, p=2.0, dim=1)  # batchsize x emb_dim
        W = F.normalize(self.W, p=2.0, dim=1)  # num_classes x emb_dim
        logits = torch.tensordot(
            x, W, dims=[[1], [1]]
        )  # batchsize x num_classes
        if labels is None:
            return logits
        with torch.no_grad():
            self.update_scale_parameter(logits, labels)
        return logits * self.s

    def update_scale_parameter(self, logits, labels, eps=1e-12):
        theta = torch.acos(torch.clamp(logits, min=-1.0 + eps, max=1.0 - eps))
        max_logits = torch.max(self.s * logits)
        B_avg = torch.where(
            labels < 1,
            torch.exp(self.s * logits - max_logits),
            torch.exp(torch.zeros_like(self.s * logits) - max_logits),
        )
        B_avg = B_avg.sum(dim=1).mean()
        theta_class = torch.sum(
            labels * theta, dim=1
        )  # take mix-upped angle of mix-upped classes
        theta_med = torch.median(theta_class)
        self.s = nn.Parameter(
            torch.clamp(
                (max_logits + torch.log(B_avg))
                / (torch.cos(torch.minimum(self.pi / 4, theta_med)) + eps),
                min=eps,
            ),
            requires_grad=False,
        )
