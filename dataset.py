# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import csv
import os

import librosa
import lightning as L
import numpy as np
import pandas as pd
import resampy
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchopenl3
from einops import rearrange
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from beats.BEATs import BEATs, BEATsConfig
from network import STFT_extractor


def adjust_size(wav, new_size: int = None):
    if new_size is None:
        return wav
    else:
        reps = int(np.ceil(new_size / wav.shape[0]))
        offset = np.random.randint(
            low=0, high=int(reps * wav.shape[0] - new_size + 1)
        )
        new_wav = np.tile(wav, reps=reps)[
            offset : offset + new_size
        ]  # randomly repeat files that are too short instead of zero-padding to increase difficulty of classification task
        return new_wav


def prep_audio(file_path: str, new_size: int = None, new_sr: int = None):
    wav, fs = sf.read(
        file_path, dtype="float32"
    )  # converting here saves a lot of memory
    wav = librosa.core.to_mono(wav.transpose()).transpose()
    if new_sr is not None:
        wav = resampy.resample(
            wav, sr_orig=16000, sr_new=48000, filter="kaiser_fast"
        )  # to speed-up the re-sampling
    wav = adjust_size(wav, new_size=new_size)
    return wav


def generate_labels(file_path: str, edition: str = "DCASE2024"):
    file = os.path.split(file_path)[1]
    if (
        edition == "DCASE2025"
        or edition == "DCASE2024"
        or edition == "DCASE2023"
        or edition == "DCASE2022"
    ):
        machine_type = os.path.split(
            os.path.split(os.path.split(file_path)[0])[0]
        )[1]
        machine_id_label = machine_type + "_" + file.split("_")[1]
        if len(file.split("_")) > 3:
            source_label = int(file.split("_")[2] == "source")
            anomaly_label = int(file.split("_")[4] == "anomaly")
            attribute_label = (
                machine_type
                + "_"
                + "_".join(file.split(".wav")[0].split("_")[6:])
            )
        else:  # handle test data of evaluation set
            if edition == "DCASE2025":
                gt_path = "./dcase2025_task2_evaluator"
            elif edition == "DCASE2024":
                gt_path = "./dcase2024_task2_evaluator"
            elif edition == "DCASE2023":
                gt_path = "./dcase2023_task2_evaluator"
            elif edition == "DCASE2022":
                gt_path = "./dcase2022_evaluator"
            gt_anomaly = dict(
                pd.read_csv(
                    os.path.join(
                        gt_path,
                        "ground_truth_data",
                        "ground_truth_"
                        + machine_type
                        + "_"
                        + file.split("_")[0]
                        + "_"
                        + file.split("_")[1]
                        + "_test.csv",
                    ),
                    header=None,
                ).to_numpy()
            )
            anomaly_label = gt_anomaly[file]
            if edition == "DCASE2022":
                gt_attribute = pd.read_csv(
                    os.path.join(
                        gt_path,
                        "ground_truth_attributes",
                        machine_type,
                        "attributes_" + file.split("_")[1] + ".csv",
                    ),
                ).to_numpy()
                # fix missing values
                for k in np.arange(gt_attribute.shape[0]):
                    if not isinstance(gt_attribute[k, -1], str):
                        j = 1
                        while str(gt_attribute[k, -j]) == "nan":
                            j = j + 1
                        gt_attribute[k, -1] = gt_attribute[k, -j]
                gt_attribute_keys = gt_attribute[:, 0]
                gt_attribute_values = gt_attribute[:, -1]
                for k in np.arange(gt_attribute.shape[0]):
                    gt_attribute_keys[k] = gt_attribute_keys[k].split("/")[-1]
                    gt_attribute_values[k] = (
                        gt_attribute_values[k].split("/")[-1].split(".wav")[0]
                    )
                gt_attribute = dict(
                    np.vstack(
                        (gt_attribute_keys, gt_attribute_values)
                    ).transpose()
                )
            else:
                gt_attribute = dict(
                    pd.read_csv(
                        os.path.join(
                            gt_path,
                            "ground_truth_attributes",
                            "ground_truth_"
                            + machine_type
                            + "_"
                            + file.split("_")[0]
                            + "_"
                            + file.split("_")[1]
                            + "_test.csv",
                        ),
                        header=None,
                    ).to_numpy()[:, :2]
                )
            source_label = int(gt_attribute[file].split("_")[2] == "source")
            attribute_label = (
                machine_type
                + "_"
                + "_".join(gt_attribute[file].split("_")[5:])
            )
    elif edition == "DCASE2020":
        machine_type = os.path.split(
            os.path.split(os.path.split(file_path)[0])[0]
        )[1]
        source_label = 1  # there is no target domain for DCASE 2020
        if len(file.split("_")) > 3:
            anomaly_label = int(file.split("_")[0] == "anomaly")
            machine_id_label = machine_type + "_" + file.split("_")[2]
        else:
            machine_id_label = machine_type + "_" + file.split("_")[1]
            with open(
                os.path.join(
                    os.path.split(
                        os.path.split(
                            os.path.split(os.path.split(file_path)[0])[0]
                        )[0]
                    )[0],
                    "eval_data_list.csv",
                )
            ) as gt_info:
                reader_obj = csv.reader(gt_info)
                found_machine = False
                for k, row in enumerate(reader_obj):
                    if row[0] == machine_type:
                        found_machine = True
                    if found_machine and row[0] == file:
                        anomaly_label = int(row[2])
                        break
        attribute_label = (
            machine_type + "_" + machine_id_label
        )  # there is no additional attribute information
    return anomaly_label, machine_id_label, source_label, attribute_label


def get_data_for_audio(
    file_path: str,
    new_size: int,
    edition: str = "DCASE2024",
    new_sr: int = None,
):
    wav = prep_audio(file_path, new_size, new_sr)
    anomaly_label, machine_id_label, source_label, attribute_label = (
        generate_labels(file_path, edition)
    )
    return wav, anomaly_label, machine_id_label, source_label, attribute_label


def get_data_from_dir(
    data_dir: str = "path/to/dcase2024/dir",
    dataset: str = "dev_data",
    data_split: str = "test",
    new_size: int = 192000,
    edition: str = "DCASE2024",
    new_sr: int = None,
    pre_model: str = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collected_wav = []
    collected_anomaly_label = []
    collected_machine_id_label = []
    collected_source_label = []
    collected_attribute_label = []
    collected_file_names = []
    if pre_model == "openL3":
        openL3_model = torchopenl3.models.load_audio_embedding_model(
            input_repr="mel128", embedding_size=512, content_type="env"
        )
    elif pre_model == "BEATs":
        checkpoint = torch.load("./BEATs_iter3.pt")
        cfg = BEATsConfig(checkpoint["cfg"])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint["model"])
        BEATs_model.eval().cuda()
    elif pre_model == "EAT":
        model_id = "worstchan/EAT-large_epoch20_pretrain"
        EAT_model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True)
            .eval()
            .cuda()
        )
        new_size = None
    elif pre_model == "STFT":
        stft = STFT_extractor(
            affine=False,
            nfft=1024,
            temporal_normalization=False,
        ).to(device)
    for category in os.listdir(os.path.join(data_dir, dataset)):
        for file_name in tqdm(
            os.listdir(os.path.join(data_dir, dataset, category, data_split))
        ):
            if file_name.endswith(".wav"):
                file_path = os.path.join(
                    data_dir, dataset, category, data_split, file_name
                )
                (
                    wav,
                    anomaly_label,
                    machine_id_label,
                    source_label,
                    attribute_label,
                ) = get_data_for_audio(file_path, new_size, edition, new_sr)
                if pre_model == "openL3":
                    with torch.no_grad():
                        wav, _ = torchopenl3.get_audio_embedding(
                            np.expand_dims(wav, axis=0),
                            sr=new_sr,
                            model=openL3_model,
                            center=True,
                            hop_size=0.1,
                            verbose=True,
                        )
                        wav = (
                            torch.mean(wav, dim=1).cpu().numpy()[0, :]
                        )  # mean pooling over temporal dimension
                elif pre_model == "BEATs":
                    with torch.no_grad():
                        wav = torch.from_numpy(np.expand_dims(wav, axis=0)).to(
                            device
                        )
                        padding_mask = torch.zeros_like(wav).bool().to(device)
                        wav = BEATs_model.extract_features(
                            wav, padding_mask=padding_mask, need_weights=True
                        )[0]
                        wav = (
                            rearrange(
                                wav[0].detach().cpu().numpy(),
                                "(t f) d -> t f d",
                                f=8,  # equals 128/16
                            )
                            .mean(axis=0)
                            .reshape(-1)
                        )  # mean pooling over temporal dimension + flattening
                elif pre_model == "EAT":
                    with torch.no_grad():
                        waveform = torch.from_numpy(wav)
                        norm_mean = -4.268
                        norm_std = 4.569
                        # Normalize and convert to mel-spectrogram
                        waveform = waveform - waveform.mean()
                        mel = torchaudio.compliance.kaldi.fbank(
                            waveform.unsqueeze(0),
                            htk_compat=True,
                            sample_frequency=16000,
                            use_energy=False,
                            window_type="hanning",
                            num_mel_bins=128,
                            dither=0.0,
                            frame_shift=10,
                        ).unsqueeze(0)

                        # Pad or truncate
                        n_frames = mel.shape[1]
                        target_length = (
                            (n_frames + 15) // 16
                        ) * 16  # Round up to nearest multiple of 16
                        if n_frames < target_length:
                            mel = torch.nn.ZeroPad2d(
                                (0, 0, 0, target_length - n_frames)
                            )(mel)
                        else:
                            reps = int(np.ceil(target_length / n_frames))
                            offset = np.random.randint(
                                low=0,
                                high=int(reps * n_frames - target_length + 1),
                            )
                            mel = mel[:, offset : offset + target_length, :]

                        # Normalize
                        mel = (mel - norm_mean) / (norm_std * 2)
                        mel = mel.unsqueeze(0).to(
                            device
                        )  # shape: [1, 1, T, F]

                        # Extract features
                        with torch.no_grad():
                            wav = EAT_model.extract_features(mel)
                        wav_seq = (
                            rearrange(
                                wav[0, 1:].detach().cpu().numpy(),
                                "(t f) d -> t f d",
                                f=8,  # equals 128/16
                            )
                            .mean(axis=0)
                            .reshape(1, -1)
                        )
                        wav = np.concatenate(
                            (wav[:, 0].detach().cpu().numpy(), wav_seq), axis=1
                        )[
                            0
                        ]  # concatenate CLS token and temporal mean of patch embeddings
                elif pre_model == "STFT":
                    wav = torch.from_numpy(np.expand_dims(wav, axis=0)).to(
                        device
                    )
                    with torch.no_grad():
                        wav = torch.abs(stft(wav))
                    wav = (
                        torch.mean(wav, dim=1).cpu().numpy()[0, :]
                    )  # mean pooling over temporal dimension
                collected_wav.append(wav)
                collected_anomaly_label.append(anomaly_label)
                collected_machine_id_label.append(machine_id_label)
                collected_source_label.append(source_label)
                collected_attribute_label.append(attribute_label)
                collected_file_names.append(
                    category
                    + "___"
                    + data_split
                    + "___"
                    + file_name.split(".wav")[0]
                )
    collected_wav = np.array(collected_wav)
    collected_anomaly_label = np.array(collected_anomaly_label)
    collected_machine_id_label = np.array(collected_machine_id_label)
    collected_source_label = np.array(collected_source_label)
    collected_attribute_label = np.array(collected_attribute_label)
    collected_file_names = np.array(collected_file_names)
    return (
        collected_wav,
        collected_anomaly_label,
        collected_machine_id_label,
        collected_source_label,
        collected_attribute_label,
        collected_file_names,
    )


def encode_labels(
    dev_train_labels, eval_train_labels, dev_test_labels, eval_test_labels
):
    le = LabelEncoder().fit(
        np.unique(
            np.concatenate(
                [
                    dev_train_labels,
                    eval_train_labels,
                    dev_test_labels,
                    eval_test_labels,
                ]
            )
        )
    )
    return (
        le.transform(dev_train_labels),
        le.transform(eval_train_labels),
        le.transform(dev_test_labels),
        le.transform(eval_test_labels),
    )


def find_max_size(data_dir: str = "path/to/dcase2024/dir"):
    max_size = -1
    for data_split in ["dev_data", "eval_data"]:
        for category in os.listdir(os.path.join(data_dir, data_split)):
            for file_name in os.listdir(
                os.path.join(data_dir, data_split, category, "train")
            ):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(
                        data_dir, data_split, category, "train", file_name
                    )
                    wav, _ = sf.read(file_path)
                    max_size = max(wav.shape[0], max_size)
                    break
    return max_size


def prepare_dataset(
    data_dir: str,
    target_dir: str,
    edition: str,
    new_sr: int = None,
    pre_model: str = None,
):
    """
    Prepare all wav files and generate/encode all labels by storing them as numpy arrays to disk.
    """
    pad_size = find_max_size(data_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # collect all partitions separately to avoid any memory issues for large datasets
    if not os.path.isfile(os.path.join(target_dir, "eval_test_wav.npy")):
        # collect data
        (
            eval_test_wav,
            eval_test_anomaly_label,
            eval_test_machine_id_label,
            eval_test_source_label,
            eval_test_attribute_label,
            eval_test_file_names,
        ) = get_data_from_dir(
            data_dir, "eval_data", "test", pad_size, edition, new_sr, pre_model
        )
        # store data
        np.save(
            os.path.join(target_dir, "eval_test_anomaly_label.npy"),
            eval_test_anomaly_label,
        )
        np.save(
            os.path.join(target_dir, "eval_test_machine_id_label.npy"),
            eval_test_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "eval_test_source_label.npy"),
            eval_test_source_label,
        )
        np.save(
            os.path.join(target_dir, "eval_test_attribute_label.npy"),
            eval_test_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "eval_test_file_names.npy"),
            eval_test_file_names,
        )
        np.save(os.path.join(target_dir, "eval_test_wav.npy"), eval_test_wav)

    if not os.path.isfile(os.path.join(target_dir, "dev_train_wav.npy")):
        # collect data
        (
            dev_train_wav,
            dev_train_anomaly_label,
            dev_train_machine_id_label,
            dev_train_source_label,
            dev_train_attribute_label,
            dev_train_file_names,
        ) = get_data_from_dir(
            data_dir, "dev_data", "train", pad_size, edition, new_sr, pre_model
        )
        # store data
        np.save(
            os.path.join(target_dir, "dev_train_anomaly_label.npy"),
            dev_train_anomaly_label,
        )
        np.save(
            os.path.join(target_dir, "dev_train_machine_id_label.npy"),
            dev_train_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "dev_train_source_label.npy"),
            dev_train_source_label,
        )
        np.save(
            os.path.join(target_dir, "dev_train_attribute_label.npy"),
            dev_train_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "dev_train_file_names.npy"),
            dev_train_file_names,
        )
        np.save(os.path.join(target_dir, "dev_train_wav.npy"), dev_train_wav)

    if not os.path.isfile(os.path.join(target_dir, "eval_train_wav.npy")):
        # collect data
        (
            eval_train_wav,
            eval_train_anomaly_label,
            eval_train_machine_id_label,
            eval_train_source_label,
            eval_train_attribute_label,
            eval_train_file_names,
        ) = get_data_from_dir(
            data_dir,
            "eval_data",
            "train",
            pad_size,
            edition,
            new_sr,
            pre_model,
        )
        np.save(
            os.path.join(target_dir, "eval_train_anomaly_label.npy"),
            eval_train_anomaly_label,
        )
        np.save(
            os.path.join(target_dir, "eval_train_machine_id_label.npy"),
            eval_train_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "eval_train_source_label.npy"),
            eval_train_source_label,
        )
        np.save(
            os.path.join(target_dir, "eval_train_attribute_label.npy"),
            eval_train_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "eval_train_file_names.npy"),
            eval_train_file_names,
        )
        np.save(os.path.join(target_dir, "eval_train_wav.npy"), eval_train_wav)

    if not os.path.isfile(os.path.join(target_dir, "dev_test_wav.npy")):
        # collect data
        (
            dev_test_wav,
            dev_test_anomaly_label,
            dev_test_machine_id_label,
            dev_test_source_label,
            dev_test_attribute_label,
            dev_test_file_names,
        ) = get_data_from_dir(
            data_dir, "dev_data", "test", pad_size, edition, new_sr, pre_model
        )
        np.save(
            os.path.join(target_dir, "dev_test_anomaly_label.npy"),
            dev_test_anomaly_label,
        )
        np.save(
            os.path.join(target_dir, "dev_test_machine_id_label.npy"),
            dev_test_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "dev_test_source_label.npy"),
            dev_test_source_label,
        )
        np.save(
            os.path.join(target_dir, "dev_test_attribute_label.npy"),
            dev_test_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "dev_test_file_names.npy"),
            dev_test_file_names,
        )
        np.save(os.path.join(target_dir, "dev_test_wav.npy"), dev_test_wav)

        # re-collect labels (inside the last if-clause to only do this once!)
        dev_train_machine_id_label = np.load(
            os.path.join(target_dir, "dev_train_machine_id_label.npy")
        )
        dev_train_attribute_label = np.load(
            os.path.join(target_dir, "dev_train_attribute_label.npy")
        )
        eval_train_machine_id_label = np.load(
            os.path.join(target_dir, "eval_train_machine_id_label.npy")
        )
        eval_train_attribute_label = np.load(
            os.path.join(target_dir, "eval_train_attribute_label.npy")
        )
        dev_test_machine_id_label = np.load(
            os.path.join(target_dir, "dev_test_machine_id_label.npy")
        )
        dev_test_attribute_label = np.load(
            os.path.join(target_dir, "dev_test_attribute_label.npy")
        )
        eval_test_machine_id_label = np.load(
            os.path.join(target_dir, "eval_test_machine_id_label.npy")
        )
        eval_test_attribute_label = np.load(
            os.path.join(target_dir, "eval_test_attribute_label.npy")
        )

        # encode labels (anomaly labels and source labels are already encoded)
        (
            dev_train_machine_id_label,
            eval_train_machine_id_label,
            dev_test_machine_id_label,
            eval_test_machine_id_label,
        ) = encode_labels(
            dev_train_machine_id_label,
            eval_train_machine_id_label,
            dev_test_machine_id_label,
            eval_test_machine_id_label,
        )
        (
            dev_train_attribute_label,
            eval_train_attribute_label,
            dev_test_attribute_label,
            eval_test_attribute_label,
        ) = encode_labels(
            dev_train_attribute_label,
            eval_train_attribute_label,
            dev_test_attribute_label,
            eval_test_attribute_label,
        )

        # store encoded labels
        np.save(
            os.path.join(target_dir, "dev_train_machine_id_label.npy"),
            dev_train_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "dev_train_attribute_label.npy"),
            dev_train_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "eval_train_machine_id_label.npy"),
            eval_train_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "eval_train_attribute_label.npy"),
            eval_train_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "dev_test_machine_id_label.npy"),
            dev_test_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "dev_test_attribute_label.npy"),
            dev_test_attribute_label,
        )
        np.save(
            os.path.join(target_dir, "eval_test_machine_id_label.npy"),
            eval_test_machine_id_label,
        )
        np.save(
            os.path.join(target_dir, "eval_test_attribute_label.npy"),
            eval_test_attribute_label,
        )
    return


def fix_filenames(data_dir: str):
    """
    fix inconsistent filenames for machine type RoboticArm (underscore is missing for attributes)
    """
    for file in os.listdir(
        os.path.join(data_dir, "eval_data", "RoboticArm", "train")
    ):
        if len(file.split("_")) < 9:
            file_path = os.path.join(
                data_dir, "eval_data", "RoboticArm", "train", file
            )
            new_file_path = os.path.join(
                data_dir,
                "eval_data",
                "RoboticArm",
                "train",
                file.split("weight")[0]
                + "weight_"
                + file.split("weight")[1].split("_")[0]
                + "_Bckg_"
                + file.split("Bckg")[1],
            )
            os.rename(file_path, new_file_path)


class DCASEASDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        use_dev_for_train: bool = True,
        use_eval_for_train: bool = True,
        use_dev_for_pred: bool = False,
        use_eval_for_pred: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_dev_for_train = use_dev_for_train
        self.use_eval_for_train = use_eval_for_train
        self.use_dev_for_pred = use_dev_for_pred
        self.use_eval_for_pred = use_eval_for_pred
        self.num_classes = self._get_num_classes()

    def _get_num_classes(self):
        dev_train_attribute_label = np.load(
            os.path.join(self.data_dir, "dev_train_attribute_label.npy")
        )
        eval_train_attribute_label = np.load(
            os.path.join(self.data_dir, "eval_train_attribute_label.npy")
        )
        train_labels = np.concatenate(
            [dev_train_attribute_label, eval_train_attribute_label], axis=0
        )
        num_classes = np.unique(train_labels).shape[0]
        return num_classes

    def setup(self, stage: str):
        if stage == "fit":
            dev_train_attribute_label = np.load(
                os.path.join(self.data_dir, "dev_train_attribute_label.npy")
            )
            dev_train_machine_id_label = torch.tensor(
                np.load(
                    os.path.join(
                        self.data_dir, "dev_train_machine_id_label.npy"
                    )
                ),
                dtype=torch.float32,
            )
            dev_train_source_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "dev_train_source_label.npy")
                ),
                dtype=torch.float32,
            )
            eval_train_attribute_label = np.load(
                os.path.join(self.data_dir, "eval_train_attribute_label.npy")
            )
            eval_train_machine_id_label = torch.tensor(
                np.load(
                    os.path.join(
                        self.data_dir, "eval_train_machine_id_label.npy"
                    )
                ),
                dtype=torch.float32,
            )
            eval_train_source_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "eval_train_source_label.npy")
                ),
                dtype=torch.float32,
            )

            train_labels = np.concatenate(
                [dev_train_attribute_label, eval_train_attribute_label], axis=0
            )
            le = LabelEncoder().fit(train_labels)
            dev_train_attribute_label = F.one_hot(
                torch.tensor(le.transform(dev_train_attribute_label)),
                num_classes=self.num_classes,
            ).float()
            eval_train_attribute_label = F.one_hot(
                torch.tensor(le.transform(eval_train_attribute_label)),
                num_classes=self.num_classes,
            ).float()
            self.train = []
            if self.use_dev_for_train:
                dev_train_wav = torch.tensor(
                    np.load(os.path.join(self.data_dir, "dev_train_wav.npy")),
                    dtype=torch.float32,
                )
                for i in range(len(dev_train_wav)):
                    if (
                        dev_train_source_label[i] == 0
                    ):  # exclude target domain when training by setting all labels to zero
                        attribute_label = torch.zeros_like(
                            dev_train_attribute_label[i]
                        )
                    else:
                        attribute_label = dev_train_attribute_label[i]
                    self.train.append(
                        [
                            dev_train_wav[i],
                            attribute_label,
                            dev_train_machine_id_label[i],
                            dev_train_source_label[i],
                        ]
                    )
            if self.use_eval_for_train:
                eval_train_wav = torch.tensor(
                    np.load(os.path.join(self.data_dir, "eval_train_wav.npy")),
                    dtype=torch.float32,
                )
                for i in range(len(eval_train_wav)):
                    if (
                        eval_train_source_label[i] == 0
                    ):  # exclude target domain when training by setting all labels to zero
                        attribute_label = torch.zeros_like(
                            eval_train_attribute_label[i]
                        )
                    else:
                        attribute_label = eval_train_attribute_label[i]
                    self.train.append(
                        [
                            eval_train_wav[i],
                            attribute_label,
                            eval_train_machine_id_label[i],
                            eval_train_source_label[i],
                        ]
                    )
            dev_test_wav = torch.tensor(
                np.load(os.path.join(self.data_dir, "dev_test_wav.npy")),
                dtype=torch.float32,
            )
            dev_test_anomaly_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "dev_test_anomaly_label.npy")
                ),
                dtype=torch.float32,
            )
            dev_test_machine_id_label = torch.tensor(
                np.load(
                    os.path.join(
                        self.data_dir, "dev_test_machine_id_label.npy"
                    )
                ),
                dtype=torch.float32,
            )
            dev_test_source_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "dev_test_source_label.npy")
                ),
                dtype=torch.float32,
            )
            self.val = []
            for i in range(len(dev_test_wav)):
                self.val.append(
                    [
                        dev_test_wav[i],
                        dev_test_anomaly_label[i],
                        dev_test_machine_id_label[i],
                        dev_test_source_label[i],
                    ]
                )
        if stage == "validate":
            dev_test_wav = torch.tensor(
                np.load(os.path.join(self.data_dir, "dev_test_wav.npy")),
                dtype=torch.float32,
            )
            dev_test_anomaly_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "dev_test_anomaly_label.npy")
                ),
                dtype=torch.float32,
            )
            dev_test_machine_id_label = torch.tensor(
                np.load(
                    os.path.join(
                        self.data_dir, "dev_test_machine_id_label.npy"
                    )
                ),
                dtype=torch.float32,
            )
            dev_test_source_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "dev_test_source_label.npy")
                ),
                dtype=torch.float32,
            )
            self.val = []
            for i in range(len(dev_test_wav)):
                self.val.append(
                    [
                        dev_test_wav[i],
                        dev_test_anomaly_label[i],
                        dev_test_machine_id_label[i],
                        dev_test_source_label[i],
                    ]
                )
        if stage == "test":
            eval_test_wav = torch.tensor(
                np.load(os.path.join(self.data_dir, "eval_test_wav.npy")),
                dtype=torch.float32,
            )
            eval_test_anomaly_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "eval_test_anomaly_label.npy")
                ),
                dtype=torch.float32,
            )
            eval_test_machine_id_label = torch.tensor(
                np.load(
                    os.path.join(
                        self.data_dir, "eval_test_machine_id_label.npy"
                    )
                ),
                dtype=torch.float32,
            )
            eval_test_source_label = torch.tensor(
                np.load(
                    os.path.join(self.data_dir, "eval_test_source_label.npy")
                ),
                dtype=torch.float32,
            )
            self.test = []
            for i in range(len(eval_test_wav)):
                self.test.append(
                    [
                        eval_test_wav[i],
                        eval_test_anomaly_label[i],
                        eval_test_machine_id_label[i],
                        eval_test_source_label[i],
                    ]
                )
        if stage == "predict":
            self.predict = []
            if self.use_dev_for_pred:
                dev_test_wav = torch.tensor(
                    np.load(os.path.join(self.data_dir, "dev_test_wav.npy")),
                    dtype=torch.float32,
                )
                dev_test_anomaly_label = torch.tensor(
                    np.load(
                        os.path.join(
                            self.data_dir, "dev_test_anomaly_label.npy"
                        )
                    ),
                    dtype=torch.float32,
                )
                dev_test_machine_id_label = torch.tensor(
                    np.load(
                        os.path.join(
                            self.data_dir, "dev_test_machine_id_label.npy"
                        )
                    ),
                    dtype=torch.float32,
                )
                dev_test_source_label = torch.tensor(
                    np.load(
                        os.path.join(
                            self.data_dir, "dev_test_source_label.npy"
                        )
                    ),
                    dtype=torch.float32,
                )
                for i in range(len(dev_test_wav)):
                    self.predict.append(
                        [
                            dev_test_wav[i],
                            dev_test_anomaly_label[i],
                            dev_test_machine_id_label[i],
                            dev_test_source_label[i],
                        ]
                    )
            if self.use_eval_for_pred:
                eval_test_wav = torch.tensor(
                    np.load(os.path.join(self.data_dir, "eval_test_wav.npy")),
                    dtype=torch.float32,
                )
                eval_test_anomaly_label = torch.tensor(
                    np.load(
                        os.path.join(
                            self.data_dir, "eval_test_anomaly_label.npy"
                        )
                    ),
                    dtype=torch.float32,
                )
                eval_test_machine_id_label = torch.tensor(
                    np.load(
                        os.path.join(
                            self.data_dir, "eval_test_machine_id_label.npy"
                        )
                    ),
                    dtype=torch.float32,
                )
                eval_test_source_label = torch.tensor(
                    np.load(
                        os.path.join(
                            self.data_dir, "eval_test_source_label.npy"
                        )
                    ),
                    dtype=torch.float32,
                )
                for i in range(len(eval_test_wav)):
                    self.predict.append(
                        [
                            eval_test_wav[i],
                            eval_test_anomaly_label[i],
                            eval_test_machine_id_label[i],
                            eval_test_source_label[i],
                        ]
                    )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
