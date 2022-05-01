import argparse
import os
from pdb import set_trace as bp

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# my lib
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from utils import to_device

from constant import THRESHOLD
from process_data import load_yaml, load_tokenizer, load_raw_data, save_pickle, load_pickle, load_json


def get_batch_data(coarse, batch_size=64):
    attrvals = []
    features = []
    for i, data in enumerate(coarse):
        for attrval in data['key_attr'].values():
            attrvals.append(attrval)
            features.append(data['feature'])
        
        if (i+1) % batch_size == 0 or (i+1) == len(coarse):
            yield attrvals, features
            attrvals = []
            features = []


def main(config, train_data_dir, text_model_name):
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    attr_config = load_json(f'{train_data_dir}/attr_config.json')
    attrval_id_map = attr_config['attrval_id_map']

    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    coarse = load_pickle(f'{train_data_dir}/coarse.pkl')
    preds = []
    with torch.no_grad():
        for attrvals, features in tqdm(get_batch_data(coarse)):

            outs = tokenizer(attrvals, padding=True)
            input_ids = outs['input_ids']
            att_mask = outs['attention_mask']

            input_ids = torch.tensor(input_ids).long().to(device)
            att_mask = torch.tensor(att_mask).bool().to(device)
            features = torch.tensor(features).float().to(device)
            
            pred = model((input_ids, att_mask, features))
            pred = list(pred.view(-1).cpu().detach().numpy())
            preds += pred
    
    i = 0
    for d in coarse:
        d['pseudo_label'] = {}
        for attrval in d['key_attr'].values():
            d['pseudo_label'][attrval] = preds[i]
            i += 1
    
    save_pickle(coarse, f'{train_data_dir}/coarse.pkl')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    args.add_argument('-t', '--train_data_dir', type=str,
                      help='train_data_dir')
    args.add_argument('-m', '--text_model_name', type=str,
                      help='text_model_name')

    config = ConfigParser.from_args(args, test=True)
    args = args.parse_args()

    main(config, args.train_data_dir, args.text_model_name)
