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
from process_data import (load_yaml, load_tokenizer, load_raw_data, 
                          save_pickle, load_pickle, load_json,
                          load_attr)


def preprocess_test_data(data_path, attr_config):
    attr_id_map = attr_config['attr_id_map']
    attrval_attr_map = attr_config['attrval_attr_map']
    attrvals = attr_config['attrvals']
    attrval_replace_map = attr_config['attrval_replace_map']

    data = load_raw_data(data_path)
    for d in data:
        d['query_map'] = {q: attr_id_map[q] for q in d['query'] if q != '图文'}

        # replace title
        for attrval, replace_val in attrval_replace_map.items():
            if attrval in d['title']:
                d['title'] = d['title'].replace(attrval, replace_val)

        # add key_attr
        d['key_attr'] = {}
        for val in attrvals:
            if val in d['title']:
                d['key_attr'][attrval_attr_map[val]] = val
    return data


if __name__ == '__main__':
    # define args
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--output_dir', type=str,
                      help='output_dir')
    args.add_argument('--test_data_path', type=str,
                      help='test_data_path')
    args.add_argument('--train_data_dir', type=str,
                      help='train_data_dir')
    args.add_argument('--text_model_name', type=str, default='bert-base-chinese',
                      help='text_model_name')
    args.add_argument('--attr_path', type=str, default='../data/contest_data/attr_to_attrvals.json',
                      help='path to attr_to_attrvals.json')

    args = args.parse_args()

    # set_device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config and state_dict
    print('loading checkpoint...')
    checkpoint = torch.load(args.resume, map_location=device)
    config = ConfigParser(checkpoint['config'], test=True)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    # load attr config
    attr_config = load_attr(args.attr_path)
    attrval_id_map = attr_config['attrval_id_map']

    # load and preprocess test data
    test_data = preprocess_test_data(args.test_data_path, attr_config)

    # build model 
    print('build model')
    pretrain_dir='/home/mw/input/pretrain_model_5238/pretrain_model'
    bert_path=f'{pretrain_dir}/model/bert-base-chinese'
    vbert_path=f'{pretrain_dir}/model/visualbert-nlvr2-coco-pre'
    attr_path = f'/home/mw/input/track1_contest_4362/train/train/attr_to_attrvals.json'
    config['arch']['args']['bert_path'] = bert_path
    config['arch']['args']['vbert_path'] = vbert_path
    config['arch']['args']['attr_path'] = attr_path

    model = config.init_obj('arch', module_arch)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # run testing
    print('run testing')
    test_output = []
    with torch.no_grad():
        for i, data in enumerate(test_data):
            title_ids = torch.tensor(
                tokenizer.encode(data['title'])).to(device)
            img_feature = torch.tensor(data['feature']).to(device)

            title_ids = title_ids.long().unsqueeze(0)
            img_feature = img_feature.float().unsqueeze(0)
            global_match = model.predict((title_ids, img_feature))

            match = {'图文': float(global_match.squeeze().cpu().detach())}
            for q, idx in data['query_map'].items():
                have_attrval = False
                for attrval, target in attrval_id_map[q].items():
                    if attrval in data['title']:
                        title_ids = torch.tensor(tokenizer.encode(
                            attrval)).to(device).unsqueeze(0)
                        have_attrval = True
                        pred = model.predict((title_ids, img_feature))
                        match[q] = float(pred.squeeze().cpu().detach())
                if not have_attrval:
                    match[q] = 0
                # match[q] = float(attr_match[:, idx].squeeze().cpu().detach())

            output = {
                'img_name': data['img_name'],
                'match': match
            }
            test_output.append(output)

    save_pickle(test_output, f'{args.output_dir}/sub.pkl')

