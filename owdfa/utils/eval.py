#!/usr/bin/env python
# coding: utf-8
import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def direct_acc(y_pred, y_true):
    """
    Calculate accuracy
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    num_correct = np.sum(y_pred == y_true)
    res = num_correct / len(y_true)
    return res


def cluster_map(y_pred, y_true):
    """
    Calculate clustering mapping relation. Require scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        row_ind, col_ind
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    _, col_ind = linear_sum_assignment(w.max() - w)
    row_ind = {col_ind[x]: x for x in col_ind}
    return row_ind, col_ind


def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scipy installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def nmi(y_pred, y_true):
    """
    Calculate normalized mutual information. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        NMI, in [0,1]
    """
    return normalized_mutual_info_score(y_true, y_pred)


def ari(y_pred, y_true):
    """
    Calculate adjusted Rand index. Require scikit-learn installed
    # Arguments
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        y_true: true labels, numpy.array with shape `(n_samples,)`
    # Return
        ARI, in [-1,1]
    """
    return adjusted_rand_score(y_true, y_pred)


def val_stat(y_tags, y_pred, y_label, y_conf):
    logger.info(f'##### Eval Results #####')
    num_known = np.max(y_label[y_tags == 1], axis=0) + 1
    known_class = y_label < num_known
    novel_class = y_label >= num_known

    mean_uncert = 1 - np.mean(y_conf)
    over_acc = cluster_acc(y_pred, y_label)
    over_nmi = nmi(y_pred, y_label)
    over_ari = ari(y_pred, y_label)
    known_acc = direct_acc(y_pred[known_class], y_label[known_class])
    known_nmi = nmi(y_pred[known_class], y_label[known_class])
    known_ari = ari(y_pred[known_class], y_label[known_class])
    if len(y_pred[novel_class]) > 0:
        novel_acc = cluster_acc(y_pred[novel_class], y_label[novel_class])
        novel_nmi = nmi(y_pred[novel_class], y_label[novel_class])
        novel_ari = ari(y_pred[novel_class], y_label[novel_class])
    else:
        novel_acc = novel_nmi = novel_ari = 0

    results = {
        'all_acc': over_acc,
        'all_nmi': over_nmi,
        'all_ari': over_ari,
        'known_acc': known_acc,
        'novel_acc': novel_acc,
        'known_nmi': known_nmi,
        'novel_nmi': novel_nmi,
        'known_ari': known_ari,
        'novel_ari': novel_ari,
        'val_uncert': mean_uncert,
    }

    logger.info(f'Uncertainty: {mean_uncert:.4f}')
    logger.info(
        f'[All] ACC: {over_acc:.4f}, NMI: {over_nmi:.4f}, ARI: {over_ari:.4f}')
    logger.info(
        f'[Known] ACC: {known_acc:.4f}, NMI: {known_nmi:.4f}, ARI: {known_ari:.4f}')
    logger.info(
        f'[Novel] ACC: {novel_acc:.4f}, NMI: {novel_nmi:.4f}, ARI: {novel_ari:.4f}')
    logger.info(f'{known_acc * 100:.2f}, '
                + f'{novel_acc * 100:.2f}, {novel_nmi * 100:.2f}, {novel_ari * 100:.2f}, '
                + f'{over_acc * 100:.2f}, {over_nmi * 100:.2f}, {over_ari * 100:.2f}')

    return results
