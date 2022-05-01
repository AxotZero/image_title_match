from pdb import set_trace as bp

from sklearn.metrics import roc_curve

import torch
import numpy as np
import torch.nn.functional as F
from constant import THRESHOLD


def binary_acc(output, target):
    if not len(output) and not len(target):
        return -1

    return torch.mean((target == (output > THRESHOLD).float()).float())


def attr_acc(output, target):
    _, attr_mask, match = target

    output = output[attr_mask]
    match = match[attr_mask]

    return binary_acc(output, match)


def global_acc(output, target):
    global_mask, _, match = target

    output = output[global_mask]
    match = match[global_mask]

    return binary_acc(output, match)


def global_f1(output, target):
    global_mask, _, match = target

    output = output[global_mask]
    y_true = match[global_mask]

    y_pred = (output > THRESHOLD).float()

    tp = (y_true * y_pred).sum().float()
    fp = ((1-y_true) * y_pred).sum().float()
    fn = (y_true * (1-y_pred)).sum().float()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return f1.item()


def attrs_acc(output, target):
    _, attrs_pred = output
    attrs_match_mask, _, attrs_label = target

    corrects = []
    for i in range(12):
        mask = attrs_match_mask[:, i]
        pred = attrs_pred[i][mask]
        label = attrs_label[:, i][mask]

        pred = torch.argmax(pred, dim=-1)

        corrects.append(pred == label)
    acc = torch.mean(torch.cat(corrects).float())
    return acc.item()


# def global_f1(output, target):
#     y_pred, _ = output
#     _, y_true, _ = target

#     tp = (y_true * y_pred).sum().float()
#     # tn = ((1-y_true) * (1-y_pred)).sum().float()
#     fp = ((1-y_true) * y_pred).sum().float()
#     fn = (y_true * (1-y_pred)).sum().float()

#     eps = 1e-7
#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)

#     f1 = 2 * (precision * recall) / (precision + recall + eps)
#     return f1.item()


# def global_acc(output, target):
#     embs_logit, _ = output
#     _, global_match, _ = target

#     return torch.mean((embs_logit == global_match).float()).item()


def val_metric(output, target):
    return (global_f1(output, target) + attrs_acc(output, target)) / 2


def attr_match_acc(output, target):
    _, attr_pred = output
    attr_match_mask, _, attr_match = target
    # mask
    attr_match_mask = attr_match_mask.view(-1)
    attr_pred = attr_pred.view(-1)
    attr_match = attr_match.view(-1)
    attr_pred = attr_pred[attr_match_mask]
    attr_match = attr_match[attr_match_mask]
    # compute
    attr_acc = torch.mean(
        (attr_match == (attr_pred > THRESHOLD).float()).float())
    return attr_acc


def global_match_acc(output, target):
    global_pred, _ = output
    _, global_match, _ = target

    # compute
    global_acc = torch.mean(
        (global_match == (global_pred > THRESHOLD).float()).float())
    return global_acc


def match_metric_wo_coarse_attr(output, target, threshold=0.5):
    return (global_match_acc(output, target) + attr_match_acc(output, target)) / 2


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def NDCG(output, target):
    with torch.no_grad():
        _, output_topk_indices = torch.topk(output, 3, dim=1)
        output_topk = torch.gather(target, 1, output_topk_indices)
        target_topk, _ = torch.topk(target, 3, dim=1)

        for i in range(3):
            output_topk[:, i] = output_topk[:, i] / logs[i]
            target_topk[:, i] = target_topk[:, i] / logs[i]
        dcg = torch.sum(output_topk, dim=1)
        idcg = torch.sum(target_topk, dim=1)
        ndcg = dcg/idcg
        ndcg = ndcg[~ndcg.isnan()]
        ndcg = torch.mean(ndcg)
    return ndcg


def NDCG16(output, target):
    return NDCG(output[:, target_indices], target[:, target_indices])


def seq2seq_drop_zero(output, target):
    with torch.no_grad():
        output = output[:, :-1]
        # set_trace()
        output = output.contiguous().view(-1, int(output.size()[-1]))
        target = target.contiguous().view(-1, int(target.size()[-1]))
        # drop all zero target
        indices = (target.sum(dim=1) != 0).type(torch.bool)
        target = target[indices]
        output = output[indices]
        return output, target


def seq2seq_NDCG(output, target):
    with torch.no_grad():
        output, target = seq2seq_drop_zero(output, target)

        return NDCG(output, target)


def seq2seq_NDCG16(output, target):
    return seq2seq_NDCG(output[:, :, target_indices], target[:, :, target_indices])
