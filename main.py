# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from argparse import ArgumentParser, Namespace
from pathlib import Path

import lightning as L
import numpy as np
import torch
import yaml

from dataset import DCASEASDDataModule, find_max_size, prepare_dataset
from lightning_module import ASDsystem, raw_system, trainedACTsystem
from utils import (
    get_ASD_performance_from_scores,
    get_mean_embs,
    print_mean_and_std_performance,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--conf-path",
        default=Path("./conf.yaml"),
        type=Path,
        help="The path to the YAML configuration file (conf.yaml).",
    )
    args = parser.parse_args()

    with open(args.conf_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
        hparams = Namespace(**params)

    dataset_path = hparams.dataset["path"]
    edition = hparams.dataset["edition"]
    mode = hparams.model["mode"]
    norm_param = str(hparams.model["norm_param"])
    if norm_param == "None" or norm_param == "none":
        norm_param = None

    # check for optional normalization parameter
    K = None
    r = None
    if norm_param is not None:
        if norm_param.isdigit():
            K = int(norm_param)
        else:
            r = float(norm_param)

    # prepare data
    if mode == "direct-act":
        target_dir = "./" + edition + "_processed"
        prepare_dataset(
            data_dir=dataset_path, target_dir=target_dir, edition=edition
        )
        input_dim = find_max_size(data_dir=dataset_path)
        max_epochs = hparams.training["num_epochs_direct-act"]
    elif mode == "openL3-act" or mode == "openL3-raw":
        target_dir = "./" + edition + "_openL3_processed"
        prepare_dataset(
            data_dir=dataset_path,
            target_dir=target_dir,
            edition=edition,
            new_sr=48000,  # required for openL3, does not need to be modified
            pre_model="openL3",
        )
        input_dim = hparams.embs["openL3_input_dim"]
        if mode == "openL3-act":
            max_epochs = hparams.training["num_epochs_openL3-act"]
        else:
            max_epochs = 0  # no training for direct usage
    elif mode == "BEATs-act" or mode == "BEATs-raw":
        target_dir = "./" + edition + "_BEATs_processed"
        prepare_dataset(
            data_dir=dataset_path,
            target_dir=target_dir,
            edition=edition,
            pre_model="BEATs",
        )
        input_dim = hparams.embs["BEATs_input_dim"]
        if mode == "BEATs-act":
            max_epochs = hparams.training["num_epochs_BEATs-act"]
        else:
            max_epochs = 0  # no training for direct usage
    else:
        raise TypeError(
            'only supported modes are "direct-act", "openL3-act", "openL3-raw", "BEATs-act" or "BEATs-raw"!'
        )
    dcase_asd = DCASEASDDataModule(
        data_dir=target_dir,
        batch_size=hparams.training["batch_size"],
        use_eval_for_train=True,
        use_dev_for_train=True,
        use_dev_for_pred=True,
        use_eval_for_pred=False,
    )
    if mode == "openL3-raw" or mode == "BEATs-raw":
        dcase_asd.setup("fit")

    n_ensemble = hparams.training["n_ensemble"]
    all_scores_eval = []
    performances_eval = np.zeros(
        (n_ensemble, 9)
    )  # there are 9 different metrics to be collected
    all_scores_dev = []
    performances_dev = np.zeros(
        (n_ensemble, 9)
    )  # there are 9 different metrics to be collected
    for k_ensemble in np.arange(n_ensemble):
        print("Running ensemble iteration #" + str(k_ensemble + 1))
        tb_logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir="logs/",
            name=edition + "_logs" + "_k_ensemble_" + str(k_ensemble),
        )

        # setup model
        if mode == "direct-act":
            asd_system = ASDsystem(
                input_dim=input_dim,
                num_classes=dcase_asd.num_classes,
                subspace_dim=hparams.model["subspace_dim"],
                bias=hparams.model["bias"],
                affine=hparams.model["affine"],
                trainable_centers=hparams.model["trainable_centers"],
                K=K,
                r=r,
            )
        elif mode == "openL3-act" or mode == "BEATs-act":
            asd_system = trainedACTsystem(
                input_dim=input_dim,
                num_classes=dcase_asd.num_classes,
                subspace_dim=hparams.model["subspace_dim"],
                bias=hparams.model["bias"],
                affine=hparams.model["affine"],
                trainable_centers=hparams.model["trainable_centers"],
                K=K,
                r=r,
            )
        elif mode == "openL3-raw" or mode == "BEATs-raw":
            asd_system = raw_system(K=K, r=r)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            logger=tb_logger,
            accelerator="gpu",
            devices=1,
        )  # no sanity checks because means are not yet available

        # train model
        if mode == "direct-act" or mode == "openL3-act" or mode == "BEATs-act":
            trainer.fit(asd_system, dcase_asd)

        # reload all data
        asd_system.eval()  # disables randomness
        # re-calculate means
        for batch_idx, batch in enumerate(dcase_asd.train_dataloader()):
            asd_system.to("cuda").reload_training_embs(batch, batch_idx)
        if mode == "direct-act" or mode == "openL3-act" or mode == "BEATs-act":
            (
                asd_system.means,
                asd_system.mean_machine_ids,
                asd_system.mean_source_labels,
            ) = get_mean_embs(
                asd_system.train_embs,
                asd_system.train_machine_ids,
                asd_system.train_source_labels,
                k=asd_system.subspace_dim,
            )

        # evaluate performance on dev data
        dcase_asd_test = DCASEASDDataModule(
            data_dir=target_dir,
            batch_size=hparams.training["batch_size"],
            use_eval_for_train=False,
            use_eval_for_pred=False,
            use_dev_for_train=True,
            use_dev_for_pred=True,
        )

        # predict with trained model and collect scores
        preds_dev = trainer.predict(asd_system, dcase_asd_test)
        scores_dev, machine_ids_dev, source_labels_dev, anomaly_labels_dev = (
            tuple(map(torch.cat, zip(*preds_dev)))
        )
        all_scores_dev.append(scores_dev)

        # compute performance for current model only
        performances_dev[k_ensemble] = np.array(
            get_ASD_performance_from_scores(
                scores_dev,
                machine_ids_dev,
                source_labels_dev,
                anomaly_labels_dev,
                print_results=True,
                return_domain_specific=True,
                eval_metrics=hparams.dataset["eval_metrics"],
            )
        )

        # evaluate performance on evaluation data
        dcase_asd_test = DCASEASDDataModule(
            data_dir=target_dir,
            batch_size=hparams.training["batch_size"],
            use_eval_for_train=True,
            use_eval_for_pred=True,
            use_dev_for_train=False,
            use_dev_for_pred=False,
        )

        # predict with trained model and collect scores
        preds_eval = trainer.predict(asd_system, dcase_asd_test)
        (
            scores_eval,
            machine_ids_eval,
            source_labels_eval,
            anomaly_labels_eval,
        ) = tuple(map(torch.cat, zip(*preds_eval)))
        all_scores_eval.append(scores_eval)

        # compute performance for current model only
        performances_eval[k_ensemble] = np.array(
            get_ASD_performance_from_scores(
                scores_eval,
                machine_ids_eval,
                source_labels_eval,
                anomaly_labels_eval,
                print_results=False,
                return_domain_specific=True,
                eval_metrics=hparams.dataset["eval_metrics"],
            )
        )
        asd_system.on_train_epoch_end()  # delete stored training embeddings

    # output performance
    print(
        "################################################################################"
    )
    print("Mean and standard deviation of results on development set")
    print(
        "################################################################################"
    )
    print_mean_and_std_performance(performances_dev)
    print(
        "################################################################################"
    )
    print("Performance of ensemble on development set")
    print(
        "################################################################################"
    )
    get_ASD_performance_from_scores(
        torch.stack(all_scores_dev).sum(dim=0),
        machine_ids_dev,
        source_labels_dev,
        anomaly_labels_dev,
        print_results=True,
        eval_metrics=hparams.dataset["eval_metrics"],
    )
    print(
        "################################################################################"
    )
    print("Mean and standard deviation of results on evaluation set")
    print(
        "################################################################################"
    )
    print_mean_and_std_performance(performances_eval)
    print(
        "################################################################################"
    )
    print("Performance of ensemble on evaluation set")
    print(
        "################################################################################"
    )
    get_ASD_performance_from_scores(
        torch.stack(all_scores_eval).sum(dim=0),
        machine_ids_eval,
        source_labels_eval,
        anomaly_labels_eval,
        print_results=True,
        eval_metrics=hparams.dataset["eval_metrics"],
    )
