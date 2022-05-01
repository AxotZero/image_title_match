import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_curve
# from metric import global_f1


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def to_device(data, device):
    if isinstance(data, tuple) or isinstance(data, list):
        return [d.to(device) for d in data]
    return data.to(device)


def f1_score(y_true, y_pred):
    tp = (y_true * y_pred).sum().float()
    # tn = ((1-y_true) * (1-y_pred)).sum().float()
    fp = ((1-y_true) * y_pred).sum().float()
    fn = (y_true * (1-y_pred)).sum().float()

    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * (precision * recall) / (precision + recall + eps)
    return f1.item()


def find_threshold(output, target):
    embs_logit, _ = output
    _, global_match, _ = target

    # detach
    embs_logit = embs_logit.detach().cpu().numpy()
    global_match = global_match.detach().cpu().numpy()

    # num_step = 10
    # min_logit = min(embs_logit)
    # max_logit = max(embs_logit)
    # step = (max_logit - min_logit) / num_step
    # cur_logit = min_logit
    # for _ in range(num_step-1):
    #     cur_logit += step
    #     y_pred = embs_logit > cur_logit
    


    fpr, tpr, thresholds = roc_curve(global_match, embs_logit) 
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


# def modify_config(config, modify_str_list):
#     """
#     modify_str_list is a list of [f'{key1}.{key2}.{key3}:value']
#     """
#     for modify_str in modify_str_list:
#         keys_str, value = modify_str.split(':')
#         keys = keys_str.split('.')
#         keys_expression = ''.join([f"[{key}]"] for key in keys)
#         eval(f"config{keys_expression}") = value