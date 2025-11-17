# Copyright (C) 2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor as LOF


def gwrp(x, r: float = 0.9, normalize: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = torch.arange(x.shape[0])
    if x.get_device() > -1:
        n = n.to(device)
    w = torch.pow(r, n)
    if normalize:
        w = w / torch.sum(w)
    return torch.sum(x * torch.unsqueeze(w, dim=1), dim=0, keepdims=True)


def get_mean_embs(
    embs,
    machine_ids,
    source_labels,
    k: int = 16,
    smote: bool = False,
    smote_num_neighbors: int = 2,
    use_mse: bool = False,
    from_lists: bool = True,
):
    means = []
    mean_source_labels = []
    mean_machine_ids = []
    if from_lists:
        embs = torch.cat(embs, dim=0)
        machine_ids = torch.cat(machine_ids, dim=0)
        source_labels = torch.cat(source_labels, dim=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for machine_id in torch.unique(machine_ids):
        if smote:
            if (
                len(np.unique(source_labels[machine_ids == machine_id].cpu()))
                == 2
            ):
                sampler = SMOTE(k_neighbors=smote_num_neighbors)
                machine_embs, target_labels = sampler.fit_resample(
                    embs[machine_ids == machine_id].cpu(),
                    source_labels[machine_ids == machine_id].cpu(),
                )
            else:
                machine_embs = (
                    embs[machine_ids == machine_id].cpu().detach().numpy()
                )
                target_labels = (
                    source_labels[machine_ids == machine_id]
                    .cpu()
                    .detach()
                    .numpy()
                )
            machine_embs = torch.from_numpy(
                machine_embs.astype(np.float32)
            ).to(device)
            if not use_mse:
                machine_embs = F.normalize(machine_embs, p=2.0, dim=1)
            machine_source_labels = torch.from_numpy(
                target_labels.astype(np.float32)
            ).to(device)
        else:
            machine_embs = embs[machine_ids == machine_id]
            machine_source_labels = source_labels[machine_ids == machine_id]
        for source in torch.unique(machine_source_labels):
            if source == 1 and k > 0:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=0,
                    n_init=10,
                    algorithm="elkan",
                    tol=1e-5,
                ).fit(
                    machine_embs[machine_source_labels == source].cpu()
                )  # slower to copy to CPU but does not require extensive GPU memory (when not using 'detach()' for embs)
                centroids = torch.from_numpy(
                    kmeans.cluster_centers_.astype(np.float32)
                ).to(device)
            else:
                centroids = machine_embs[
                    machine_source_labels == source
                ]  # just use all samples of the target domain
            # centroids = F.normalize(centroids, p=2.0, dim=1)  # degrades performance, don't do this
            means.append(centroids)
            mean_machine_ids.append(
                torch.ones(centroids.size()[0]).to(device) * machine_id
            )
            mean_source_labels.append(
                torch.ones(centroids.size()[0]).to(device) * source
            )
    return (
        torch.cat(means, dim=0),
        torch.cat(mean_machine_ids, dim=0),
        torch.cat(mean_source_labels, dim=0),
    )


def get_standardization_stats(
    mean_embs,
    mean_machine_ids,
    mean_source_labels,
    test_embs,
    test_machine_ids,
    use_mse: bool = False,
    K: int = None,
    r: float = None,
    norm_type: str = "ratio",
):
    test_embs = torch.cat(test_embs, dim=0)
    test_machine_ids = torch.cat(test_machine_ids, dim=0)
    source_scores = get_scores(
        mean_embs[mean_source_labels > 0.5],
        mean_machine_ids[mean_source_labels > 0.5],
        mean_source_labels[mean_source_labels > 0.5],
        test_embs,
        test_machine_ids,
        use_mse=use_mse,
        K=K,
        r=r,
        norm_type=norm_type,
    )
    target_scores = get_scores(
        mean_embs[mean_source_labels < 0.5],
        mean_machine_ids[mean_source_labels < 0.5],
        mean_source_labels[mean_source_labels < 0.5],
        test_embs,
        test_machine_ids,
        use_mse=use_mse,
        K=K,
        r=r,
        norm_type=norm_type,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_machine_ids = torch.max(test_machine_ids).int() + 1
    dists_source_mean = torch.zeros(num_machine_ids).to(device)
    dists_source_std = torch.ones(num_machine_ids).to(device)
    dists_target_mean = torch.zeros(num_machine_ids).to(device)
    dists_target_std = torch.ones(num_machine_ids).to(device)
    for machine_id in torch.unique(test_machine_ids):
        if torch.sum(mean_machine_ids == machine_id) > 0:
            dists_source_mean[machine_id.int()] = torch.mean(
                source_scores[test_machine_ids == machine_id]
            )
            dists_source_std[machine_id.int()] = torch.std(
                source_scores[test_machine_ids == machine_id]
            )
            dists_target_mean[machine_id.int()] = torch.mean(
                target_scores[test_machine_ids == machine_id]
            )
            dists_target_std[machine_id.int()] = torch.std(
                target_scores[test_machine_ids == machine_id]
            )
    return (
        dists_source_mean,
        dists_source_std,
        dists_target_mean,
        dists_target_std,
    )


def get_pairwise_distance(x, y, use_mse: bool = False):
    if use_mse:
        return 0.25 * torch.cdist(x, y)
    else:
        return 0.5 * (1 - torch.mm(x, torch.transpose(y, 0, 1)))


def get_scores(
    mean_embs,
    mean_machine_ids,
    mean_source_labels,  # required for domain-wise standardization
    test_embs,
    test_machine_ids,
    use_mse: bool = False,
    K: int = None,
    r: float = None,
    norm_type: str = "ratio",
    standardize: bool = False,
    dists_source_mean=None,
    dists_source_std=None,
    dists_target_mean=None,
    dists_target_std=None,
):
    scores = torch.ones_like(test_machine_ids) * float("Inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for machine_id in torch.unique(test_machine_ids):
        if torch.sum(mean_machine_ids == machine_id) > 0:
            if norm_type == "LOF":
                lof = LOF(n_neighbors=K, novelty=True)
                lof.fit(mean_embs[(mean_machine_ids == machine_id)].cpu())
                scores[test_machine_ids == machine_id] = -torch.from_numpy(
                    lof.score_samples(
                        test_embs[test_machine_ids == machine_id].cpu()
                    ).astype(np.float32)
                ).to(device)
            else:
                score_mod = 1
                if K is not None or r is not None:
                    ref_scores = torch.sort(
                        get_pairwise_distance(
                            mean_embs[mean_machine_ids == machine_id],
                            mean_embs[mean_machine_ids == machine_id],
                            use_mse,
                        ),
                        dim=0,
                    ).values
                    if K is not None:
                        score_mod = torch.mean(
                            ref_scores[1 : K + 1], dim=0, keepdims=True
                        )
                    elif r is not None:
                        score_mod = gwrp(
                            ref_scores[1:], r=r, normalize=True
                        )  # for additive normalization, 'normalize' should be set to 'True'
                dists = get_pairwise_distance(
                    test_embs[test_machine_ids == machine_id],
                    mean_embs[mean_machine_ids == machine_id],
                    use_mse,
                )
                if norm_type == "ratio":
                    dists = dists / score_mod
                elif norm_type == "difference":
                    dists = dists - score_mod
                if standardize:
                    dists_source = dists[
                        :,
                        (
                            mean_source_labels[mean_machine_ids == machine_id]
                            > 0.5
                        ),
                    ]
                    dists_target = dists[
                        :,
                        (
                            mean_source_labels[mean_machine_ids == machine_id]
                            < 0.5
                        ),
                    ]
                    dists[
                        :,
                        (
                            mean_source_labels[mean_machine_ids == machine_id]
                            > 0.5
                        ),
                    ] = (
                        dists_source - dists_source_mean[machine_id.int()]
                    ) / dists_source_std[
                        machine_id.int()
                    ]
                    dists[
                        :,
                        (
                            mean_source_labels[mean_machine_ids == machine_id]
                            < 0.5
                        ),
                    ] = (
                        dists_target - dists_target_mean[machine_id.int()]
                    ) / dists_target_std[
                        machine_id.int()
                    ]
                scores[test_machine_ids == machine_id] = torch.min(
                    dists, dim=1
                ).values
    return scores


def get_ASD_performance_from_scores(
    scores,
    test_machine_ids,
    test_source_labels,
    test_anomaly_labels,
    print_results: bool = False,
    return_domain_specific: bool = False,
    eval_metrics: str = "official",
):
    # compute performance metrics
    aucs = []
    aucs_source = []
    aucs_target = []
    paucs = []
    paucs_source = []
    paucs_target = []
    auroc = torchmetrics.AUROC(task="binary")
    pauroc = torchmetrics.AUROC(
        task="binary", max_fpr=0.1
    )  # as used in the DCASE Challenge
    for machine_id in torch.unique(test_machine_ids):
        for source in torch.unique(test_source_labels):
            if eval_metrics == "official":
                auc = auroc(
                    scores[
                        (test_machine_ids == machine_id)
                        & (
                            (test_source_labels == source)
                            | (test_anomaly_labels == 1)
                        )
                    ],
                    test_anomaly_labels[
                        (test_machine_ids == machine_id)
                        & (
                            (test_source_labels == source)
                            | (test_anomaly_labels == 1)
                        )
                    ],
                )
            elif eval_metrics == "simple":
                auc = auroc(
                    scores[
                        (test_machine_ids == machine_id)
                        & (test_source_labels == source)
                    ],
                    test_anomaly_labels[
                        (test_machine_ids == machine_id)
                        & (test_source_labels == source)
                    ],
                )
            else:
                raise TypeError(
                    'only supported evaluation metrics are "official" or "simple"!'
                )
            pauc = pauroc(
                scores[
                    (test_machine_ids == machine_id)
                    * (test_source_labels == source)
                ],
                test_anomaly_labels[
                    (test_machine_ids == machine_id)
                    * (test_source_labels == source)
                ],
            )
            aucs.append(auc)
            if int(source.item()) > 0.5:
                aucs_source.append(auc)
                paucs_source.append(pauc)
            else:
                aucs_target.append(auc)
                paucs_target.append(pauc)
            if print_results:
                print(
                    "AUC for machine id "
                    + str(machine_id.item())
                    + ", "
                    + str(["target", "source"][int(source.item())])
                    + " domain: "
                    + str(np.round(auc.item() * 100, 1))
                )
                print(
                    "pAUC for machine id "
                    + str(machine_id.item())
                    + ", "
                    + str(["target", "source"][int(source.item())])
                    + " domain: "
                    + str(np.round(pauc.item() * 100, 1))
                )
        pauc = pauroc(
            scores[test_machine_ids == machine_id],
            test_anomaly_labels[test_machine_ids == machine_id],
        )
        paucs.append(pauc)
        if print_results:
            if torch.unique(test_source_labels).numel() > 1:
                print(
                    "pAUC for machine id "
                    + str(machine_id.item())
                    + ", joint domain: "
                    + str(np.round(pauc.item() * 100, 1))
                )
            print(
                "--------------------------------------------------------------------------------"
            )
    amean_auc = torch.mean(torch.stack(aucs))
    amean_pauc = torch.mean(torch.stack(paucs))
    amauc = torch.mean(
        torch.cat((torch.stack(aucs), torch.stack(paucs)), dim=0)
    )
    hmean_auc = harmonic_mean(torch.stack(aucs))
    hmean_pauc = harmonic_mean(torch.stack(paucs))
    hmauc = harmonic_mean(
        torch.cat((torch.stack(aucs), torch.stack(paucs)), dim=0)
    )
    hmean_auc_source = harmonic_mean(torch.stack(aucs_source))
    hmean_pauc_source = harmonic_mean(torch.stack(paucs_source))
    hmauc_source = harmonic_mean(
        torch.cat((torch.stack(aucs_source), torch.stack(paucs_source)), dim=0)
    )
    if torch.unique(test_source_labels).numel() > 1:
        hmean_auc_target = harmonic_mean(torch.stack(aucs_target))
        hmean_pauc_target = harmonic_mean(torch.stack(paucs_target))
        hmauc_target = harmonic_mean(
            torch.cat(
                (torch.stack(aucs_target), torch.stack(paucs_target)), dim=0
            )
        )
    else:
        return_domain_specific = False
    if print_results:
        if torch.unique(test_source_labels).numel() > 1:
            print(
                "################################################################################"
            )
            print(
                "harmonic mean of AUCs for source domain: "
                + str(np.round(hmean_auc_source.item() * 100, 1))
            )
            print(
                "harmonic mean of pAUCs for source domain: "
                + str(np.round(hmean_pauc_source.item() * 100, 1))
            )
            print(
                "harmonic mean of AUCs and pAUCs for source domain: "
                + str(np.round(hmauc_source.item() * 100, 1))
            )
            print(
                "################################################################################"
            )
            print(
                "harmonic mean of AUCs for target domain: "
                + str(np.round(hmean_auc_target.item() * 100, 1))
            )
            print(
                "harmonic mean of pAUCs for target domain: "
                + str(np.round(hmean_pauc_target.item() * 100, 1))
            )
            print(
                "harmonic mean of AUCs and pAUCs for target domain: "
                + str(np.round(hmauc_target.item() * 100, 1))
            )
        print(
            "################################################################################"
        )
        print(
            "arithmetic mean of AUCs: "
            + str(np.round(amean_auc.item() * 100, 1))
        )
        print(
            "arithmetic mean of pAUCs: "
            + str(np.round(amean_pauc.item() * 100, 1))
        )
        print(
            "arithmetic mean of AUCs and pAUCs: "
            + str(np.round(amauc.item() * 100, 1))
        )
        print(
            "################################################################################"
        )
        print(
            "harmonic mean of AUCs: "
            + str(np.round(hmean_auc.item() * 100, 1))
        )
        print(
            "harmonic mean of pAUCs: "
            + str(np.round(hmean_pauc.item() * 100, 1))
        )
        print(
            "harmonic mean of AUCs and pAUCs: "
            + str(np.round(hmauc.item() * 100, 1))
        )
        print(
            "################################################################################"
        )
    if return_domain_specific:
        return (
            hmean_auc_source,
            hmean_pauc_source,
            hmauc_source,
            hmean_auc_target,
            hmean_pauc_target,
            hmauc_target,
            hmean_auc,
            hmean_pauc,
            hmauc,
        )
    else:
        return amean_auc, amean_pauc, amauc, hmean_auc, hmean_pauc, hmauc


def get_ASD_performance(
    mean_embs,
    mean_machine_ids,
    mean_source_labels,
    test_embs,
    test_machine_ids,
    test_source_labels,
    test_anomaly_labels,
    print_results: bool = False,
    return_domain_specific: bool = False,
    use_mse: bool = False,
    K: int = None,
    r: float = None,
    norm_type: str = "ratio",
    standardize: bool = False,
    dists_source_mean=None,
    dists_source_std=None,
    dists_target_mean=None,
    dists_target_std=None,
):
    test_embs = torch.cat(test_embs, dim=0)
    test_machine_ids = torch.cat(test_machine_ids, dim=0)
    test_source_labels = torch.cat(test_source_labels, dim=0)
    test_anomaly_labels = torch.cat(test_anomaly_labels, dim=0)
    scores = get_scores(
        mean_embs,
        mean_machine_ids,
        mean_source_labels,
        test_embs,
        test_machine_ids,
        use_mse=use_mse,
        K=K,
        r=r,
        norm_type=norm_type,
        standardize=standardize,
        dists_source_mean=dists_source_mean,
        dists_source_std=dists_source_std,
        dists_target_mean=dists_target_mean,
        dists_target_std=dists_target_std,
    )
    return get_ASD_performance_from_scores(
        scores,
        test_machine_ids,
        test_source_labels,
        test_anomaly_labels,
        print_results,
        return_domain_specific,
    )


def harmonic_mean(x, keepdims: bool = False):
    return ((1 / x).mean(dim=0, keepdims=keepdims)) ** (-1)


def print_mean_and_std_performance(
    performance_list, return_domain_specific=False
):
    if return_domain_specific:
        print("source domain (AUC, pAUC, hmean):")
    else:
        print("arithmetic mean over sections (AUC, pAUC, amean):")
    print(
        [
            str(m) + " +/- " + str(n)
            for m, n in zip(
                np.round(np.mean(performance_list * 100, axis=0), 1)[:3],
                np.round(np.std(performance_list * 100, axis=0), 1)[:3],
            )
        ]
    )
    if return_domain_specific:
        print("target domain (AUC, pAUC, hmean):")
    else:
        print("harmonic mean over sections (AUC, pAUC, hmean):")
    print(
        [
            str(m) + " +/- " + str(n)
            for m, n in zip(
                np.round(np.mean(performance_list * 100, axis=0), 1)[3:6],
                np.round(np.std(performance_list * 100, axis=0), 1)[3:6],
            )
        ]
    )
    if return_domain_specific:
        print("mixed domain (AUC, pAUC, hmean):")
        print(
            [
                str(m) + " +/- " + str(n)
                for m, n in zip(
                    np.round(np.mean(performance_list * 100, axis=0), 1)[6:9],
                    np.round(np.std(performance_list * 100, axis=0), 1)[6:9],
                )
            ]
        )
    return
