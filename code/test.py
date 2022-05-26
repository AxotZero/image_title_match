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

from constant import THRESHOLD, RELATION_VISUAL_MASK
from process_data import (load_yaml, load_tokenizer, load_raw_data, 
                          save_pickle, load_pickle, load_json,
                          load_attr)


def run_stat(sub):
    stat = {}
    for d in sub:
        for k, v in d['match'].items():
            if k not in stat:
                stat[k] = {0:0, 1:0}
            stat[k][v>0.5] += 1
    return stat


def preprocess_test_data(data_path, attr_config):
    attr_id_map = attr_config['attr_id_map']
    attrval_attr_map = attr_config['attrval_attr_map']
    attrvals = attr_config['attrvals']
    attrval_replace_map = attr_config['attrval_replace_map']
    attrval_id_map = attr_config['attrval_id_map']

    data = load_raw_data(data_path)
    for d in data:
        # replace title
        for attrval, replace_val in attrval_replace_map.items():
            if attrval in d['title']:
                d['title'] = d['title'].replace(attrval, replace_val)

        query_map = {}
        for q in d['query']:
            if q == '图文':
                continue
            
            has_attr = False
            for attrval in attrval_id_map[q].keys():
                if attrval in d['title']:
                    has_attr = True
                    query_map[q] = attrval
                    
            if not has_attr:
                raise ValueError
        d['query_map'] = query_map


        key_attr = {}
        for val in attrvals:
            if val in d['title']:
                key_attr[attrval_attr_map[val]] = val
        
        if ('裤门襟' in key_attr) and ('闭合方式' in key_attr):
            if '裤型' in key_attr or '裤长' in key_attr:
                key_attr.pop('闭合方式')
            elif '鞋帮高度' in key_attr:
                key_attr.pop('裤门襟')
            # else:
            #     key_attr.pop('闭合方式')
            #     key_attr.pop('裤门襟')

        d['key_attr'] = key_attr

    return data


def get_batch_data(datas, attr_config, batch_size=64):
    texts = []
    img_features = []
    visual_masks = []

    for i, data in enumerate(datas):
        feature = data['feature']

        # create global data
        # create global visual mask
        global_visual_mask = [0] * 13
        for attr in data['key_attr'].keys():
            global_visual_mask[attr_config['attr_id_map'][attr]] = 1
        global_visual_mask[-1] = 1

        texts.append(data['title'])
        img_features.append(feature)
        visual_masks.append(global_visual_mask)

        # create attr data
        for attr, attrval in data['query_map'].items():
            # attr visual mask
            visual_mask = [0] * 13
            visual_mask[attr_config['attr_id_map'][attr]] = 1
            
            texts.append(attrval)
            img_features.append(feature)
            visual_masks.append(visual_mask)
        
        if (i+1) % batch_size == 0 or (i == len(datas)-1):
            yield (texts, img_features, visual_masks)

            texts = []
            img_features = []
            visual_masks = []


def relation_visual_mask_data(datas, attr_config, batch_size=64):
    texts = []
    img_features = []
    visual_masks = []

    for i, data in enumerate(datas):
        feature = data['feature']

        # create global data
        # create global visual mask

        global_visual_mask = np.array([0] * 12)
        for attr in data['key_attr'].keys():
            global_visual_mask = global_visual_mask | RELATION_VISUAL_MASK[attr_config['attr_id_map'][attr]]
        global_visual_mask = list(global_visual_mask)
        global_visual_mask.append(1)

        texts.append(data['title'])
        img_features.append(feature)
        visual_masks.append(global_visual_mask)

        # create attr data
        for attr, attrval in data['query_map'].items():
            # attr visual mask
            visual_mask = list(RELATION_VISUAL_MASK[attr_config['attr_id_map'][attr]])
            visual_mask.append(0)
            
            texts.append(attrval)
            img_features.append(feature)
            visual_masks.append(visual_mask)
        
        if (i+1) % batch_size == 0 or (i == len(datas)-1):
            yield (texts, img_features, visual_masks)

            texts = []
            img_features = []
            visual_masks = []



def all_in_one_batch_data(datas, attr_config, batch_size=64):
    texts = []
    img_features = []
    visual_masks = []
    query = []

    for i, data in enumerate(datas):
        feature = data['feature']

        # create global data
        # create global visual mask
        global_visual_mask = [0] * 13
        for attr in data['key_attr'].keys():
            global_visual_mask[attr_config['attr_id_map'][attr]] = 1
        global_visual_mask[-1] = 1

        texts.append(data['title'])
        img_features.append(feature)
        visual_masks.append(global_visual_mask)
        query.append(data['query'])
        
        
        if (i+1) % batch_size == 0 or (i == len(datas)-1):
            yield (texts, img_features, visual_masks, query)

            texts = []
            img_features = []
            visual_masks = []
            query = []


def parse_args():
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--output_path', type=str,
                      help='output_path')
    args.add_argument('--test_data_path', type=str,
                      help='test_data_path')
    args.add_argument('--train_data_dir', type=str,
                      help='train_data_dir')
    args.add_argument('--text_model_name', type=str, default='bert-base-chinese',
                      help='text_model_name')
    args.add_argument('--bert_path', type=str, default='../data/contest_data/attr_to_attrvals.json',
                      help='path to attr_to_attrvals.json')
    args.add_argument('--vbert_path', type=str, default='../data/contest_data/attr_to_attrvals.json',
                      help='path to attr_to_attrvals.json')
    args.add_argument('--attr_path', type=str, default='../data/contest_data/attr_to_attrvals.json',
                      help='path to attr_to_attrvals.json')
    args.add_argument('--all_in_one', action='store_true',
                      help='all_in_one')
    args.add_argument('--relation_mask', action='store_true',
                      help='all_in_one')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    # define args
    args = parse_args()

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
    config['arch']['args']['bert_path'] = args.bert_path
    config['arch']['args']['vbert_path'] = args.vbert_path
    config['arch']['args']['attr_path'] = args.attr_path

    model = config.init_obj('arch', module_arch)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    # run testing
    print('run testing')
    preds = []
    with torch.no_grad():
        if args.all_in_one:
            pbar = tqdm(enumerate(all_in_one_batch_data(test_data, attr_config)))
            for i, (texts, features, visual_masks, query) in pbar:
                outs = tokenizer(texts, padding=True)
                texts_ids = outs['input_ids']
                text_masks = outs['attention_mask']

                texts_ids = torch.tensor(texts_ids).to(device).long()
                features = torch.tensor(features).to(device).float()
                text_masks = torch.tensor(text_masks).to(device).bool()
                visual_masks = torch.tensor(visual_masks).to(device).bool()

                # apply query
                pred = model((texts_ids, features, text_masks, visual_masks))
                pred = pred.cpu().detach().numpy().tolist()
                for qs, p in zip(query, pred):
                    for q in qs:
                        if q == '图文':
                            preds.append(p[-1])
                        else:
                            preds.append(p[attr_config['attr_id_map'][q]])
        else:
            if args.relation_mask:
                pbar = tqdm(enumerate(relation_visual_mask_data(test_data, attr_config)))
            else:
                pbar = tqdm(enumerate(get_batch_data(test_data, attr_config)))
            for i, (texts, features, visual_masks) in pbar:
                outs = tokenizer(texts, padding=True)
                texts_ids = outs['input_ids']
                text_masks = outs['attention_mask']

                texts_ids = torch.tensor(texts_ids).to(device).long()
                features = torch.tensor(features).to(device).float()
                text_masks = torch.tensor(text_masks).to(device).bool()
                visual_masks = torch.tensor(visual_masks).to(device).bool()

                pred = model((texts_ids, features, text_masks, visual_masks))
                pred = list(pred.view(-1).cpu().detach().numpy())
                preds += pred


    # map to sub
    i = 0
    test_output = []
    for d in test_data:
        
        match = {}
        for q in d['query']:
            match[q] = preds[i]
            i += 1

        test_output.append({
            'img_name': d['img_name'],
            'match': match
        })
    save_pickle(test_output, args.output_path)

    print(run_stat(test_output))