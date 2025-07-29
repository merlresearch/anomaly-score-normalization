# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from network import FFT_branch, STFT_branch, STFT_extractor, shallow_classifier

torch.manual_seed(0)


def test_classifier_output_shape():
    # shallow classifier
    input_dim, emb_dim = 512, 128
    model = shallow_classifier(input_dim=input_dim, emb_dim=emb_dim)
    nbatch = 16
    input = torch.rand((nbatch, input_dim))
    embeddings = model(input)
    expected_shape_embeddings = (nbatch, emb_dim)
    assert embeddings.shape == expected_shape_embeddings


def test_stft_extractor_output_shape():
    model = STFT_extractor()
    nbatch = 16
    input_dim = 192000
    input = torch.rand((nbatch, input_dim))
    specs = model(input)
    time_dim = 374
    freq_dim = 513
    expected_shape_specs = (nbatch, time_dim, freq_dim)
    assert specs.shape == expected_shape_specs


def test_stft_branch_output_shape():
    model = STFT_branch()
    nbatch = 16
    time_dim = 374
    freq_dim = 513
    input = torch.rand((nbatch, time_dim, freq_dim))
    embeddings = model(input)
    emb_dim = 256
    expected_shape_embeddings = (nbatch, emb_dim)
    assert embeddings.shape == expected_shape_embeddings


def test_fft_branch_output_shape():
    input_dim = 192000
    model = FFT_branch(input_dim=input_dim)
    nbatch = 16
    input = torch.rand((nbatch, input_dim))
    embeddings = model(input)
    emb_dim = 256
    expected_shape_embeddings = (nbatch, emb_dim)
    assert embeddings.shape == expected_shape_embeddings
