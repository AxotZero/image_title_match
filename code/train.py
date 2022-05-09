import os
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

from ranger import Ranger



# fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# # torch.backends.cudnn.enabled = False
# # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# np.random.seed(SEED)
# os.environ['TOKENIZERS_PARALLELISM'] = "false"

def set_seed(seed):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['TOKENIZERS_PARALLELISM'] = "false"


def main(config):
    logger = config.get_logger('train')

    set_seed(config['seed'])

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer = Ranger(model.parameters(), **config['optimizer']['args'])
    lr_scheduler = None

    trainer = Trainer(model, criterion, metrics, optimizer,
                       config=config,
                       device=device,
                       device_ids=device_ids,
                       data_loader=data_loader,
                       valid_data_loader=None,
                       lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="0", type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float, target='optimizer;args;lr'),
        CustomArgs(['--ep', '--epochs'],
                   type=int, target='trainer;epochs'),
        CustomArgs(['--bs', '--batch_size'], type=int,
                   target='data_loader;args;batch_size'),
        CustomArgs(['--fid', '--fold_idx'], type=int,
                   target='data_loader;args;fold_idx'),
        CustomArgs(['--loss', '--loss_func'], type=str, target='loss'),
        CustomArgs(['--save_dir'], type=str,
                   target='trainer;save_dir'),
        CustomArgs(['--bert_path'], type=str,
                   target='arch;args;bert_path'),
        CustomArgs(['--vbert_path'], type=str,
                   target='arch;args;vbert_path'),
        CustomArgs(['--tokenizer_path'], type=str,
                   target='data_loader;args;tokenizer_path'),
        CustomArgs(['--raw_data_dir'], type=str,
                   target='data_loader;args;raw_data_dir'),
        CustomArgs(['--data_length'], type=int,
                   target='data_loader;args;length'),
        
    ]
    config = ConfigParser.from_args(args, options=options)
    # os.system(f"cp {args.parse_args().config} {config['trainer']['save_dir']}")
    main(config)
