# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataset import generate_labels


def test_dataset_parsing_2023():
    edition = "DCASE2023"
    file_path = "./DCASE2023_dataset/dev_data/fan/train/section_00_source_train_normal_0017_m-n_W.wav"
    anomaly_label, machine_id_label, source_label, attribute_label = (
        generate_labels(file_path, edition)
    )
    assert anomaly_label == 0
    assert machine_id_label == "fan_00"
    assert source_label == 1
    assert attribute_label == "fan_m-n_W"


def test_dataset_parsing_2024():
    edition = "DCASE2024"
    file_path = "./DCASE2024_dataset/dev_data/toothbrush/train/section_00_target_train_normal_0007_noAttribute.wav"
    anomaly_label, machine_id_label, source_label, attribute_label = (
        generate_labels(file_path, edition)
    )
    assert anomaly_label == 0
    assert machine_id_label == "toothbrush_00"
    assert source_label == 0
    assert attribute_label == "toothbrush_noAttribute"


def test_dataset_parsing_2020():
    edition = "DCASE2020"
    file_path = (
        "./DCASE2020_dataset/dev_data/pump/test/anomaly_id_06_00000063.wav"
    )
    anomaly_label, machine_id_label, source_label, attribute_label = (
        generate_labels(file_path, edition)
    )
    assert anomaly_label == 1
    assert machine_id_label == "pump_06"
    assert source_label == 1
    assert attribute_label == "pump_pump_06"
