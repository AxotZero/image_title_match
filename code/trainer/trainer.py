import numpy as np
import torch
from torchvision.utils import make_grid
from pdb import set_trace as bp

from tqdm import tqdm

from base import BaseTrainer
from utils import inf_loop, MetricTracker, to_device, find_threshold


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, device_ids,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, device_ids, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.data_loader.dataset.training = True
        # val_log = self._valid_epoch(epoch)
        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=self.len_epoch, ncols=70) as pbar:
            for batch_idx, (data, target) in enumerate(self.data_loader):
                
                try:
                    data, target = to_device(
                        data, self.device), to_device(target, self.device)

                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()

                    self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                    self.train_metrics.update('loss', loss.item())
                    for met in self.metric_ftns:
                        self.train_metrics.update(
                            met.__name__, met(output, target))

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory, skipping batch')
                        self.optimizer.zero_grad()
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad
                        torch.cuda.empty_cache()
                    else:
                        print(e)
                        raise
                pbar.set_description(
                    f"Epoch: {epoch} Loss: {loss.item():.5f}"
                )
                pbar.update()

                torch.cuda.empty_cache()
                self.optimizer.zero_grad()


                if batch_idx == self.len_epoch:
                    break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.valid_data_loader.dataset.training = False
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = to_device(
                    data, self.device), to_device(target, self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target))

            # threshold = find_threshold(outputs, targets)
            # self.model.update_threshold(threshold)
            # print(f' === got threshold { self.model.threshold.item() : 0.4f}===')
            # outputs[0] = (outputs[0] >= threshold).float()

            # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx+1, 'valid')
            # self.valid_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.valid_metrics.update(met.__name__, met(outputs, targets))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def cat_all(self, data):
        num_obj_per_data = len(data[0])
        objs = [[] for _ in range(num_obj_per_data)]
        for d in data:
            for i in range(num_obj_per_data):
                objs[i].append(d[i])

        for i in range(num_obj_per_data):
            # if isinstance(objs[i][0], tuple) or isinstance(objs[i][0], list):
            #     # attr_outpus: [[[] * 14], [[] * 14]] -> [[] * 14]
            #     objs[i] = self.cat_all(objs[i])
            # else:
            objs[i] = torch.cat(objs[i])

        return objs

    def to_device(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            return (to_device(d) for d in data)
        return data.to(self.device)

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
