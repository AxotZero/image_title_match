from pdb import set_trace as bp
# from attr import dataclass
import random
import copy
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pickle
import pandas as pd
import numpy as np

from base import BaseDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from process_data import load_attr, load_raw_data

import jieba


def given_prob(prob):
    if random.uniform(0, 1) < prob:
        return True
    return False


def preprocess_train_data(data_path, attr_config, is_fine=False, length=-1):
    """
    1. replace similar attrval in title to primary attrval
    2. key_attr
        if is_fine:
            replace the key_attr to primary attrval
        else:
            add key_attr wich is the attrval in title
    """
    "# 高领/半高领、松紧/松紧带、拉链、系带"

    data = load_raw_data(data_path, length=length)

    attrvals = attr_config['attrvals']
    attrval_attr_map = attr_config['attrval_attr_map']
    attrval_replace_map = attr_config['attrval_replace_map']

    for d in data:
        d['is_fine'] = is_fine

        # replace title key_attr
        for attrval, replace_val in attrval_replace_map.items():
            if attrval in d['title']:
                d['title'] = d['title'].replace(attrval, replace_val)
        
        # replace key_attr
        if is_fine:
            for k, v in d['key_attr'].items():
                d['key_attr'][k] = attrval_replace_map[d['key_attr'][k]]
        
        # add key_attr
        else:
            key_attr = {}
            for val in attrvals:
                if val in d['title']:
                    key_attr[attrval_attr_map[val]] = val
            
            if ('裤门襟' in key_attr) and ('闭合方式' in key_attr):
                if '裤型' in key_attr or '裤长' in key_attr:
                    key_attr.pop('闭合方式')
                elif '鞋帮高度' in key_attr:
                    key_attr.pop('裤门襟')
                else:
                    key_attr.pop('闭合方式')
                    key_attr.pop('裤门襟')

            d['key_attr'] = key_attr


    return data
    # save_pickle(data, f'{save_dir}/{save_name}')


def add_attrvals_to_jieba(attrvals):
    for word in attrvals:
        jieba.add_word(word)


class HardDataLoader(BaseDataLoader):
    def batch_collate(data):
        titles_ids = []
        features = []
        matches = []
        global_masks = []
        attr_masks = []

        for d in data:
            for title_ids in d['text_ids']:
                titles_ids += [torch.tensor(title_ids)]
            features += d['feature']
            matches += d['match']
            global_masks += d['global_mask']
            attr_masks += d['attr_mask']

        titles_ids = nn.utils.rnn.pad_sequence(
            titles_ids, batch_first=True, padding_value=0)
        # titles_mask = (titles_ids != 0).long()
        features = torch.tensor(features)
        matches = torch.tensor(matches)
        global_masks = torch.tensor(global_masks)
        attr_masks = torch.tensor(attr_masks)

        return (
            # (titles_ids.long(), titles_mask.long(), features.float()),
            (titles_ids.long(), features.float()),
            (global_masks.bool(), attr_masks.bool(), matches.float())
        )

    class InnerDataset(Dataset):
        def __init__(self, 
                     raw_data_dir,
                     tokenizer_path,
                     length=-1,
                     run_jieba=False):
            self.attr_config = load_attr(f'{raw_data_dir}/attr_to_attrvals.json')

            fine = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_fine.txt', 
                attr_config=self.attr_config, 
                is_fine=True
            )
            coarse = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_coarse.txt', 
                attr_config=self.attr_config, 
                is_fine=False
            )

            self.data = fine + coarse
                
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.mask_token = self.tokenizer.mask_token
            self.run_jieba = run_jieba
            add_attrvals_to_jieba(self.attr_config['attrvals'])

            self.training = True

            self.global_match1 = 1
            self.global_match0 = 1
            self.attr_match1 = 1
            self.attr_match0 = 1
                
        def __len__(self):
            return len(self.data)

        def replace_attr(self, text, attr, attrval):
            num_classes = self.attr_config['attr_num_classes'][attr]
            cur_idx = self.attr_config['attrval_id_map'][attr][attrval]
            add_idx = int(random.uniform(1, num_classes))
            replace_val = self.attr_config['classid_attrval_map'][attr][(cur_idx+add_idx) % num_classes]
            return text.replace(attrval, replace_val)

        @property
        def global_match0_prob(self):
            return self.global_match1 / (self.global_match0 + self.global_match1)

        @property
        def attr_match0_prob(self):
            return self.attr_match1 / (self.attr_match0 + self.attr_match1)

        def __getitem__(self, index):
            """
            1. title data aug
                - 以0.1的機率，將title換成別的title，並將match設為 0
                - 以0.4的機率，將title 的 attrval 換成別的 attrval， 並將match設為 0
            2. 若你是 fine ， 每個 attrval of key_attr 有 0.5 的機率會被替換成別的 attrval，並將 match 設為0
            3. 將 title 與 attrval 都產生 text_ids, 並在 batch 的維度上 align
            4. 產生 global_mask 跟 attr_mask 
            5. 產生的 data 的 length = len of (data['match'])，包含
                - text_ids (title or attr_val): (a list of list(int))
                - feature (都是同一個feature):   (a list of 2048-dim float, all are the same)
                - is_match:                     (a list of is_match) 
                - global_mask                   (where is the 圖文 match)
                - attr_mask                     (where are attr match)
            """
            attr_config = self.attr_config
            data = self.data[index].copy()

            ret = {
                'text_ids': [],
                'feature': [],
                'match': [],
                'global_mask': [],
                'attr_mask': [],
                # 'attention_mask': []
            }

            # mismatch attrvals
            neg_attrvals = copy.deepcopy(attr_config['attrvals'])
            for v in data['key_attr'].values():
                neg_attrvals.remove(v)

            ### global match
            match = data['match']['图文']
            ret['global_mask'].append(1)
            ret['attr_mask'].append(0)
            text = data['title']
            
            # run jieba data aug
            if self.training and self.run_jieba and given_prob(0.3):
                words = list(jieba.cut(text))
                random.shuffle(words)
                text = ''.join(words)

            # key_attr augment 
            if self.training:
                attrs = list(data['key_attr'].keys())
                if len(attrs) >= 1:
                    attr = random.choice(attrs)
                    attrval = data['key_attr'][attr]

                    # generate neg sample
                    if match and given_prob(self.global_match0_prob):
                        match = 0
                        rand2 = random.uniform(0, 1)
                        # replace with same attr attrval
                        # if rand2 < 0.8:
                        text = self.replace_attr(text, attr, attrval)
                        
                        # add attrval
                        # elif 0.8 < rand2 < 1.0:
                        #     words = list(jieba.cut(text))
                        #     insert_index = random.choice(range(len(words)))
                        #     add_attrval = random.choice(neg_attrvals)
                        #     words = words[:insert_index] + [add_attrval] + words[insert_index:]
                        #     text = ''.join(words)

                    # delete key_attr
                    # elif not match and given_prob(0.3):
                    #     text = text.replace(attrval, '')



                # delete one attr no effect for match
                # elif len(attrs) >= 2 and given_prob(0.5):
                #     delete_attr = random.choice(attrs)
                #     text = text.replace(data['key_attr'][delete_attr], '')
                    # text = self.replace_attr(text, delete_attr, )
            # elif self.training and match == 
            
            # update global match rate
            if self.training:
                self.global_match0 += (match == 0)
                self.global_match1 += (match == 1)

            ret['match'].append(match)
            ret['feature'].append(data['feature'])
            
            text_ids = self.tokenizer.encode(text)
            ret['text_ids'].append(text_ids)
            
            ### attr match
            if data['is_fine'] or self.training:
                for k, text in data['key_attr'].items():
                    match = data['match']['图文']
                    if not data['is_fine'] and match == 0:
                        continue

                    ret['global_mask'].append(0)
                    ret['attr_mask'].append(1)

                    if self.training and match == 1 and given_prob(self.attr_match0_prob):
                        match = 0

                        # replace with random attrval of same attr
                        # if given_prob(0.8):
                        cur_idx = self.attr_config['attrval_id_map'][k][text]
                        num_classes = attr_config['attr_num_classes'][k]
                        add_idx = int(random.uniform(1, num_classes))
                        text = attr_config['classid_attrval_map'][k][(cur_idx+add_idx) % num_classes]

                        # text = k + self.tokenizer.sep_token + text

                        # replace with neg_attrvals
                        # else:
                        #     text = random.choice(neg_attrvals)

                    if self.training:
                        self.attr_match0 += (match == 0)
                        self.attr_match1 += (match == 1)

                    ret['match'].append(match)
                    ret['feature'].append(data['feature'])
                    ret['text_ids'].append(self.tokenizer.encode(text))

            return ret

    def __init__(self, raw_data_dir, tokenizer_path, length=-1, run_jieba=False,
                batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.25, num_workers=1, training=True):
        self.dataset = self.__class__.InnerDataset(
            raw_data_dir=raw_data_dir, 
            tokenizer_path=tokenizer_path, 
            length=length,
            run_jieba=run_jieba
        )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            fold_idx,
            validation_split,
            num_workers,
            collate_fn=self.__class__.batch_collate)


class VisualMaskDataLoader(BaseDataLoader):
    def batch_collate(data):
        titles_ids = []
        features = []
        matches = []
        global_masks = []
        attr_masks = []
        visual_masks = []

        for d in data:
            for title_ids in d['text_ids']:
                titles_ids += [torch.tensor(title_ids)]
            features += d['feature']
            matches += d['match']
            global_masks += d['global_mask']
            attr_masks += d['attr_mask']
            visual_masks += d['visual_mask']

        titles_ids = nn.utils.rnn.pad_sequence(
            titles_ids, batch_first=True, padding_value=0)
        features = torch.tensor(features)
        matches = torch.tensor(matches)
        global_masks = torch.tensor(global_masks)
        attr_masks = torch.tensor(attr_masks)
        visual_masks = torch.tensor(visual_masks)

        return (
            (titles_ids.long(), features.float(), visual_masks.bool()),
            (global_masks.bool(), attr_masks.bool(), matches.float())
        )

    class InnerDataset(Dataset):
        def __init__(self, 
                     raw_data_dir,
                     tokenizer_path,
                     run_jieba=False,
                     mask_attr=True,
                     mask_global=False,
                     length=-1):
            self.attr_config = load_attr(f'{raw_data_dir}/attr_to_attrvals.json')

            fine = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_fine.txt', 
                attr_config=self.attr_config, 
                is_fine=True
            )
            coarse = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_coarse.txt', 
                attr_config=self.attr_config, 
                is_fine=False
            )

            self.data = fine + coarse
                
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.run_jieba = run_jieba
            add_attrvals_to_jieba(self.attr_config['attrvals'])

            self.training = True
            self.mask_attr = mask_attr
            self.mask_global = mask_global
            self.global_match1 = 1
            self.global_match0 = 1
            self.attr_match1 = 1
            self.attr_match0 = 1
                
        def __len__(self):
            return len(self.data)

        def replace_attr(self, text, attr, attrval):
            num_classes = self.attr_config['attr_num_classes'][attr]
            cur_idx = self.attr_config['attrval_id_map'][attr][attrval]
            add_idx = int(random.uniform(1, num_classes))
            replace_val = self.attr_config['classid_attrval_map'][attr][(cur_idx+add_idx) % num_classes]
            return text.replace(attrval, replace_val)

        @property
        def global_match0_prob(self):
            return self.global_match1 / (self.global_match0 + self.global_match1)

        @property
        def attr_match0_prob(self):
            return self.attr_match1 / (self.attr_match0 + self.attr_match1)

        def __getitem__(self, index):
            """
            1. title data aug
                - 以0.1的機率，將title換成別的title，並將match設為 0
                - 以0.4的機率，將title 的 attrval 換成別的 attrval， 並將match設為 0
            2. 若你是 fine ， 每個 attrval of key_attr 有 0.5 的機率會被替換成別的 attrval，並將 match 設為0
            3. 將 title 與 attrval 都產生 text_ids, 並在 batch 的維度上 align
            4. 產生 global_mask 跟 attr_mask 
            5. 產生的 data 的 length = len of (data['match'])，包含
                - text_ids (title or attr_val): (a list of list(int))
                - feature (都是同一個feature):   (a list of 2048-dim float, all are the same)
                - is_match:                     (a list of is_match) 
                - global_mask                   (where is the 圖文 match)
                - attr_mask                     (where are attr match)
            """
            attr_config = self.attr_config
            attr_id_map = attr_config['attr_id_map']
            attrval_id_map = attr_config['attrval_id_map']

            data = self.data[index].copy()

            ret = {
                'text_ids': [],
                'feature': [],
                'match': [],
                'global_mask': [],
                'attr_mask': [],
                'visual_mask': []
            }

            # mismatch attrvals
            neg_attrvals = copy.deepcopy(attr_config['attrvals'])
            for v in data['key_attr'].values():
                neg_attrvals.remove(v)

            ### global match
            match = data['match']['图文']
            text = data['title']
            
            # run jieba data aug
            if self.training and self.run_jieba and given_prob(0.3):
                words = list(jieba.cut(text))
                random.shuffle(words)
                text = ''.join(words)

            # augment global match
            if self.training:
                attrs = list(data['key_attr'].keys())
                if len(attrs) >= 1:
                    attr = random.choice(attrs)
                    attrval = data['key_attr'][attr]

                    # generate neg sample
                    if match and given_prob(self.global_match0_prob):
                        match = 0
                        # rand2 = random.uniform(0, 1)
                        text = self.replace_attr(text, attr, attrval)
                        
                        
            # update global match rate
            if self.training:
                self.global_match0 += (match == 0)
                self.global_match1 += (match == 1)
            
            ret['global_mask'].append(1)
            ret['attr_mask'].append(0)
            ret['match'].append(match)
            ret['feature'].append(data['feature'])
            ret['text_ids'].append(self.tokenizer.encode(text))

            # visual mask
            if self.mask_global:
                visual_mask = [0] * 13
                for attr in data['key_attr'].keys():
                    visual_mask[attr_id_map[attr]] = 1
                visual_mask[-1] = 1
                ret['visual_mask'].append(visual_mask)
            else:
                ret['visual_mask'].append([1]*13)
            
            ### attr match
            if data['is_fine'] or self.training:
                for k, text in data['key_attr'].items():
                    match = data['match']['图文']
                    if not data['is_fine'] and match == 0:
                        continue

                    if self.training and match == 1 and given_prob(self.attr_match0_prob):
                        match = 0

                        cur_idx = self.attr_config['attrval_id_map'][k][text]
                        num_classes = attr_config['attr_num_classes'][k]
                        add_idx = int(random.uniform(1, num_classes))
                        text = attr_config['classid_attrval_map'][k][(cur_idx+add_idx) % num_classes]

                    if self.training:
                        self.attr_match0 += (match == 0)
                        self.attr_match1 += (match == 1)

                    ret['global_mask'].append(0)
                    ret['attr_mask'].append(1)
                    ret['match'].append(match)
                    ret['feature'].append(data['feature'])
                    ret['text_ids'].append(self.tokenizer.encode(text))

                    # visual mask
                    if self.mask_attr:
                        visual_mask = [0] * 13
                        visual_mask[attr_id_map[k]] = 1
                        ret['visual_mask'].append(visual_mask)
                    else:
                        ret['visual_mask'].append([1]*13)

            return ret

    def __init__(self, raw_data_dir, tokenizer_path, length=-1, run_jieba=False,
                mask_attr=True, mask_global=False,
                batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.25, num_workers=1, training=True):
        self.dataset = self.__class__.InnerDataset(
            raw_data_dir=raw_data_dir, tokenizer_path=tokenizer_path, length=length, run_jieba=run_jieba, mask_attr=mask_attr, mask_global=mask_global,)

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            fold_idx,
            validation_split,
            num_workers,
            collate_fn=self.__class__.batch_collate)



class AttrMaskImageEnhanceDataLoader(BaseDataLoader):
    def batch_collate(data):
        titles_ids = []
        features = []
        matches = []
        global_masks = []
        attr_masks = []
        attr_classify_masks = []
        attr_classify_labels = []
        visual_masks = []

        for d in data:
            for title_ids in d['text_ids']:
                titles_ids += [torch.tensor(title_ids)]
            features += d['feature']
            matches += d['match']
            global_masks += d['global_mask']
            attr_masks += d['attr_mask']
            attr_classify_masks += d['attr_classify_mask']
            attr_classify_labels += d['attr_classify_label']
            visual_masks += d['visual_mask']

        titles_ids = nn.utils.rnn.pad_sequence(
            titles_ids, batch_first=True, padding_value=0)
        features = torch.tensor(features)
        matches = torch.tensor(matches)
        global_masks = torch.tensor(global_masks)
        attr_masks = torch.tensor(attr_masks)
        attr_classify_masks = torch.tensor(attr_classify_masks)
        attr_classify_labels = torch.tensor(attr_classify_labels)
        visual_masks = torch.tensor(visual_masks)

        return (
            (titles_ids.long(), features.float(), visual_masks.bool()),
            (global_masks.bool(), attr_masks.bool(), matches.float(), attr_classify_masks.bool(), attr_classify_labels.long())
        )

    class InnerDataset(Dataset):
        def __init__(self, 
                     raw_data_dir,
                     tokenizer_path,
                     run_jieba=False,
                     length=-1):
            self.attr_config = load_attr(f'{raw_data_dir}/attr_to_attrvals.json')

            fine = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_fine.txt', 
                attr_config=self.attr_config, 
                is_fine=True
            )
            coarse = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_coarse.txt', 
                attr_config=self.attr_config, 
                is_fine=False
            )

            self.data = fine + coarse
                
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.run_jieba = run_jieba
            add_attrvals_to_jieba(self.attr_config['attrvals'])

            self.training = True

            self.global_match1 = 1
            self.global_match0 = 1
            self.attr_match1 = 1
            self.attr_match0 = 1
                
        def __len__(self):
            return len(self.data)

        def replace_attr(self, text, attr, attrval):
            num_classes = self.attr_config['attr_num_classes'][attr]
            cur_idx = self.attr_config['attrval_id_map'][attr][attrval]
            add_idx = int(random.uniform(1, num_classes))
            replace_val = self.attr_config['classid_attrval_map'][attr][(cur_idx+add_idx) % num_classes]
            return text.replace(attrval, replace_val)

        @property
        def global_match0_prob(self):
            return self.global_match1 / (self.global_match0 + self.global_match1)

        @property
        def attr_match0_prob(self):
            return self.attr_match1 / (self.attr_match0 + self.attr_match1)

        def __getitem__(self, index):
            """
            1. title data aug
                - 以0.1的機率，將title換成別的title，並將match設為 0
                - 以0.4的機率，將title 的 attrval 換成別的 attrval， 並將match設為 0
            2. 若你是 fine ， 每個 attrval of key_attr 有 0.5 的機率會被替換成別的 attrval，並將 match 設為0
            3. 將 title 與 attrval 都產生 text_ids, 並在 batch 的維度上 align
            4. 產生 global_mask 跟 attr_mask 
            5. 產生的 data 的 length = len of (data['match'])，包含
                - text_ids (title or attr_val): (a list of list(int))
                - feature (都是同一個feature):   (a list of 2048-dim float, all are the same)
                - is_match:                     (a list of is_match) 
                - global_mask                   (where is the 圖文 match)
                - attr_mask                     (where are attr match)
            """
            attr_config = self.attr_config
            attr_id_map = attr_config['attr_id_map']
            attrval_id_map = attr_config['attrval_id_map']

            data = self.data[index].copy()

            ret = {
                'text_ids': [],
                'feature': [],
                'match': [],
                'global_mask': [],
                'attr_mask': [],
                'attr_classify_mask': [],
                'attr_classify_label': [],
                'visual_mask': []
            }

            base_attr_classify_mask = [0] * 12
            base_attr_classify_label = [-1] * 12
            # base_visual_mask = []

            # mismatch attrvals
            neg_attrvals = copy.deepcopy(attr_config['attrvals'])
            for v in data['key_attr'].values():
                neg_attrvals.remove(v)

            ### global match
            match = data['match']['图文']
            ret['global_mask'].append(1)
            ret['attr_mask'].append(0)
            text = data['title']
            
            # generate img enhance target
            attr_classify_mask = deepcopy(base_attr_classify_mask)
            attr_classify_label = deepcopy(base_attr_classify_label)
            if match == 1:
                for attr, attrval in data['key_attr'].items():
                    attr_classify_mask[attr_id_map[attr]] = 1
                    label_idx = attrval_id_map[attr][attrval]  # 2
                    attr_classify_label[attr_id_map[attr]] = label_idx
            ret['attr_classify_label'].append(attr_classify_label)
            ret['attr_classify_mask'].append(attr_classify_mask)
            ret['visual_mask'].append([1] * 13)

            # run jieba data aug
            if self.training and self.run_jieba and given_prob(0.3):
                words = list(jieba.cut(text))
                random.shuffle(words)
                text = ''.join(words)


            # augment global match
            if self.training:
                attrs = list(data['key_attr'].keys())
                if len(attrs) >= 1:
                    attr = random.choice(attrs)
                    attrval = data['key_attr'][attr]

                    # generate neg sample
                    if match and given_prob(self.global_match0_prob):
                        match = 0
                        rand2 = random.uniform(0, 1)
                        text = self.replace_attr(text, attr, attrval)
                        
                        
            # update global match rate
            if self.training:
                self.global_match0 += (match == 0)
                self.global_match1 += (match == 1)

            ret['match'].append(match)
            ret['feature'].append(data['feature'])
            
            text_ids = self.tokenizer.encode(text)
            ret['text_ids'].append(text_ids)
            
            ### attr match
            if data['is_fine'] or self.training:
                for k, text in data['key_attr'].items():
                    match = data['match']['图文']
                    if not data['is_fine'] and match == 0:
                        continue

                    ret['global_mask'].append(0)
                    ret['attr_mask'].append(1)

                    if self.training and match == 1 and given_prob(self.attr_match0_prob):
                        match = 0

                        cur_idx = self.attr_config['attrval_id_map'][k][text]
                        num_classes = attr_config['attr_num_classes'][k]
                        add_idx = int(random.uniform(1, num_classes))
                        text = attr_config['classid_attrval_map'][k][(cur_idx+add_idx) % num_classes]

                    if self.training:
                        self.attr_match0 += (match == 0)
                        self.attr_match1 += (match == 1)

                    ret['match'].append(match)
                    ret['feature'].append(data['feature'])
                    ret['text_ids'].append(self.tokenizer.encode(text))
                    ret['attr_classify_label'].append(deepcopy(base_attr_classify_label))
                    ret['attr_classify_mask'].append(deepcopy(base_attr_classify_mask))

                    visual_mask = [0] * 13
                    visual_mask[attr_id_map[k]] = 1
                    ret['visual_mask'].append(visual_mask)

            return ret

    def __init__(self, raw_data_dir, tokenizer_path, length=-1, run_jieba=False,
                batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.25, num_workers=1, training=True):
        self.dataset = self.__class__.InnerDataset(
            raw_data_dir=raw_data_dir, tokenizer_path=tokenizer_path, length=length, run_jieba=run_jieba)

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            fold_idx,
            validation_split,
            num_workers,
            collate_fn=self.__class__.batch_collate)




class ImageEnhanceDataLoader(BaseDataLoader):
    def batch_collate(data):
        titles_ids = []
        features = []
        matches = []
        global_masks = []
        attr_masks = []
        attr_classify_masks = []
        attr_classify_labels = []

        for d in data:
            for title_ids in d['text_ids']:
                titles_ids += [torch.tensor(title_ids)]
            features += d['feature']
            matches += d['match']
            global_masks += d['global_mask']
            attr_masks += d['attr_mask']
            attr_classify_masks += d['attr_classify_mask']
            attr_classify_labels += d['attr_classify_label']

        titles_ids = nn.utils.rnn.pad_sequence(
            titles_ids, batch_first=True, padding_value=0)
        # titles_mask = (titles_ids != 0).long()
        features = torch.tensor(features)
        matches = torch.tensor(matches)
        global_masks = torch.tensor(global_masks)
        attr_masks = torch.tensor(attr_masks)
        attr_classify_masks = torch.tensor(attr_classify_masks)
        attr_classify_labels = torch.tensor(attr_classify_labels)

        return (
            (titles_ids.long(), features.float()),
            (global_masks.bool(), attr_masks.bool(), matches.float(), attr_classify_masks.bool(), attr_classify_labels.long())
        )

    class InnerDataset(Dataset):
        def __init__(self, 
                     raw_data_dir,
                     tokenizer_path,
                     run_jieba=False,
                     length=-1):
            self.attr_config = load_attr(f'{raw_data_dir}/attr_to_attrvals.json')

            fine = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_fine.txt', 
                attr_config=self.attr_config, 
                is_fine=True
            )
            coarse = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_coarse.txt', 
                attr_config=self.attr_config, 
                is_fine=False
            )

            self.data = fine + coarse
                
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.run_jieba = run_jieba
            add_attrvals_to_jieba(self.attr_config['attrvals'])

            self.training = True

            self.global_match1 = 1
            self.global_match0 = 1
            self.attr_match1 = 1
            self.attr_match0 = 1
                
        def __len__(self):
            return len(self.data)

        def replace_attr(self, text, attr, attrval):
            num_classes = self.attr_config['attr_num_classes'][attr]
            cur_idx = self.attr_config['attrval_id_map'][attr][attrval]
            add_idx = int(random.uniform(1, num_classes))
            replace_val = self.attr_config['classid_attrval_map'][attr][(cur_idx+add_idx) % num_classes]
            return text.replace(attrval, replace_val)

        @property
        def global_match0_prob(self):
            return self.global_match1 / (self.global_match0 + self.global_match1)

        @property
        def attr_match0_prob(self):
            return self.attr_match1 / (self.attr_match0 + self.attr_match1)

        def __getitem__(self, index):
            """
            1. title data aug
                - 以0.1的機率，將title換成別的title，並將match設為 0
                - 以0.4的機率，將title 的 attrval 換成別的 attrval， 並將match設為 0
            2. 若你是 fine ， 每個 attrval of key_attr 有 0.5 的機率會被替換成別的 attrval，並將 match 設為0
            3. 將 title 與 attrval 都產生 text_ids, 並在 batch 的維度上 align
            4. 產生 global_mask 跟 attr_mask 
            5. 產生的 data 的 length = len of (data['match'])，包含
                - text_ids (title or attr_val): (a list of list(int))
                - feature (都是同一個feature):   (a list of 2048-dim float, all are the same)
                - is_match:                     (a list of is_match) 
                - global_mask                   (where is the 圖文 match)
                - attr_mask                     (where are attr match)
            """
            attr_config = self.attr_config
            attr_id_map = attr_config['attr_id_map']
            attrval_id_map = attr_config['attrval_id_map']

            data = self.data[index].copy()

            ret = {
                'text_ids': [],
                'feature': [],
                'match': [],
                'global_mask': [],
                'attr_mask': [],
                'attr_classify_mask': [],
                'attr_classify_label': []
            }

            base_attr_classify_mask = [0] * 12
            base_attr_classify_label = [-1] * 12

            # mismatch attrvals
            neg_attrvals = copy.deepcopy(attr_config['attrvals'])
            for v in data['key_attr'].values():
                neg_attrvals.remove(v)

            ### global match
            match = data['match']['图文']
            ret['global_mask'].append(1)
            ret['attr_mask'].append(0)
            text = data['title']
            
            # generate img enhance target
            attr_classify_mask = deepcopy(base_attr_classify_mask)
            attr_classify_label = deepcopy(base_attr_classify_label)
            if match == 1:
                for attr, attrval in data['key_attr'].items():
                    attr_classify_mask[attr_id_map[attr]] = 1
                    label_idx = attrval_id_map[attr][attrval]  # 2
                    attr_classify_label[attr_id_map[attr]] = label_idx
            ret['attr_classify_label'].append(attr_classify_label)
            ret['attr_classify_mask'].append(attr_classify_mask)

            # run jieba data aug
            if self.training and self.run_jieba and given_prob(0.3):
                words = list(jieba.cut(text))
                random.shuffle(words)
                text = ''.join(words)


            # augment global match
            if self.training:
                attrs = list(data['key_attr'].keys())
                if len(attrs) >= 1:
                    attr = random.choice(attrs)
                    attrval = data['key_attr'][attr]

                    # generate neg sample
                    if match and given_prob(self.global_match0_prob):
                        match = 0
                        rand2 = random.uniform(0, 1)
                        # replace with same attr attrval
                        # if rand2 < 0.8:
                        text = self.replace_attr(text, attr, attrval)
                        
                        # add attrval
                        # elif 0.8 < rand2 < 1.0:
                        #     words = list(jieba.cut(text))
                        #     insert_index = random.choice(range(len(words)))
                        #     add_attrval = random.choice(neg_attrvals)
                        #     words = words[:insert_index] + [add_attrval] + words[insert_index:]
                        #     text = ''.join(words)

                    # delete key_attr
                    # elif not match and given_prob(0.3):
                    #     text = text.replace(attrval, '')



                # delete one attr no effect for match
                # elif len(attrs) >= 2 and given_prob(0.5):
                #     delete_attr = random.choice(attrs)
                #     text = text.replace(data['key_attr'][delete_attr], '')
                    # text = self.replace_attr(text, delete_attr, )
            # elif self.training and match == 
            
            # update global match rate
            if self.training:
                self.global_match0 += (match == 0)
                self.global_match1 += (match == 1)

            ret['match'].append(match)
            ret['feature'].append(data['feature'])
            
            text_ids = self.tokenizer.encode(text)
            ret['text_ids'].append(text_ids)
            
            ### attr match
            if data['is_fine'] or self.training:
                for k, text in data['key_attr'].items():
                    match = data['match']['图文']
                    if not data['is_fine'] and match == 0:
                        continue

                    ret['global_mask'].append(0)
                    ret['attr_mask'].append(1)

                    if self.training and match == 1 and given_prob(self.attr_match0_prob):
                        match = 0

                        # replace with random attrval of same attr
                        # if given_prob(0.8):
                        cur_idx = self.attr_config['attrval_id_map'][k][text]
                        num_classes = attr_config['attr_num_classes'][k]
                        add_idx = int(random.uniform(1, num_classes))
                        text = attr_config['classid_attrval_map'][k][(cur_idx+add_idx) % num_classes]

                        # text = k + self.tokenizer.sep_token + text

                        # replace with neg_attrvals
                        # else:
                        #     text = random.choice(neg_attrvals)

                    if self.training:
                        self.attr_match0 += (match == 0)
                        self.attr_match1 += (match == 1)

                    ret['match'].append(match)
                    ret['feature'].append(data['feature'])
                    ret['text_ids'].append(self.tokenizer.encode(text))
                    ret['attr_classify_label'].append(deepcopy(base_attr_classify_label))
                    ret['attr_classify_mask'].append(deepcopy(base_attr_classify_mask))

            return ret

    def __init__(self, raw_data_dir, tokenizer_path, length=-1, run_jieba=False,
                batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.25, num_workers=1, training=True):
        self.dataset = self.__class__.InnerDataset(
            raw_data_dir=raw_data_dir, tokenizer_path=tokenizer_path, length=length, run_jieba=run_jieba)

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            fold_idx,
            validation_split,
            num_workers,
            collate_fn=self.__class__.batch_collate)



class MyDataLoader(BaseDataLoader):
    def batch_collate(data):
        titles_ids = []
        features = []
        matches = []
        global_masks = []
        attr_masks = []

        for d in data:
            for title_ids in d['text_ids']:
                titles_ids += [torch.tensor(title_ids)]
            features += d['feature']
            matches += d['match']
            global_masks += d['global_mask']
            attr_masks += d['attr_mask']

        titles_ids = nn.utils.rnn.pad_sequence(
            titles_ids, batch_first=True, padding_value=0)
        # titles_mask = (titles_ids != 0).long()
        features = torch.tensor(features)
        matches = torch.tensor(matches)
        global_masks = torch.tensor(global_masks)
        attr_masks = torch.tensor(attr_masks)

        return (
            # (titles_ids.long(), titles_mask.long(), features.float()),
            (titles_ids.long(), features.float()),
            (global_masks.bool(), attr_masks.bool(), matches.float())
        )

    class InnerDataset(Dataset):
        def __init__(self, 
                     raw_data_dir,
                     tokenizer_path,
                     length=-1,
                     run_jieba=False):
            self.attr_config = load_attr(f'{raw_data_dir}/attr_to_attrvals.json')

            fine = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_fine.txt', 
                attr_config=self.attr_config, 
                is_fine=True
            )
            coarse = preprocess_train_data(
                length=length,
                data_path=f'{raw_data_dir}/train_coarse.txt', 
                attr_config=self.attr_config, 
                is_fine=False
            )

            self.data = fine + coarse
                
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.mask_token = self.tokenizer.mask_token
            self.run_jieba = run_jieba
            add_attrvals_to_jieba(self.attr_config['attrvals'])

            self.training = True

            self.global_match1 = 1
            self.global_match0 = 1
            self.attr_match1 = 1
            self.attr_match0 = 1
                
        def __len__(self):
            return len(self.data)

        def replace_attr(self, text, attr, attrval):
            num_classes = self.attr_config['attr_num_classes'][attr]
            cur_idx = self.attr_config['attrval_id_map'][attr][attrval]
            add_idx = int(random.uniform(1, num_classes))
            replace_val = self.attr_config['classid_attrval_map'][attr][(cur_idx+add_idx) % num_classes]
            return text.replace(attrval, replace_val)

        @property
        def global_match0_prob(self):
            return self.global_match1 / (self.global_match0 + self.global_match1)

        @property
        def attr_match0_prob(self):
            return self.attr_match1 / (self.attr_match0 + self.attr_match1)

        def __getitem__(self, index):
            """
            1. title data aug
                - 以0.1的機率，將title換成別的title，並將match設為 0
                - 以0.4的機率，將title 的 attrval 換成別的 attrval， 並將match設為 0
            2. 若你是 fine ， 每個 attrval of key_attr 有 0.5 的機率會被替換成別的 attrval，並將 match 設為0
            3. 將 title 與 attrval 都產生 text_ids, 並在 batch 的維度上 align
            4. 產生 global_mask 跟 attr_mask 
            5. 產生的 data 的 length = len of (data['match'])，包含
                - text_ids (title or attr_val): (a list of list(int))
                - feature (都是同一個feature):   (a list of 2048-dim float, all are the same)
                - is_match:                     (a list of is_match) 
                - global_mask                   (where is the 圖文 match)
                - attr_mask                     (where are attr match)
            """
            attr_config = self.attr_config
            data = self.data[index].copy()

            ret = {
                'text_ids': [],
                'feature': [],
                'match': [],
                'global_mask': [],
                'attr_mask': [],
                # 'attention_mask': []
            }

            # mismatch attrvals
            neg_attrvals = copy.deepcopy(attr_config['attrvals'])
            for v in data['key_attr'].values():
                neg_attrvals.remove(v)

            ### global match
            match = data['match']['图文']
            ret['global_mask'].append(1)
            ret['attr_mask'].append(0)
            text = data['title']
            
            # run jieba data aug
            if self.training and self.run_jieba and given_prob(0.5):
                words = list(jieba.cut(text))

                # random mask word
                if given_prob(0.5):
                    for i, w in enumerate(words):
                        if w not in attr_config['attrvals'] and given_prob(0.1):
                            words[i] = self.mask_token
                
                # random shuffle
                if given_prob(0.5):
                    random.shuffle(words)

                text = ''.join(words)

            # key_attr augment 
            if self.training and match == 1:
                attrs = list(data['key_attr'].keys())
                # min_replace_length = 1 if data['is_fine'] else 2
                min_replace_length = 1
                ## replace key_attr
                if len(attrs) >= min_replace_length and given_prob(self.global_match0_prob):
                    match = 0
                    replace_attrs = random.sample(attrs, min_replace_length)
                    for attr in replace_attrs:
                        text = self.replace_attr(text, attr, data['key_attr'][attr])

                    for attr, attrval in data['key_attr'].items():
                        if attr in replace_attrs:
                            continue

                        rand2 = random.uniform(0, 1)
                        # replace with same attr attrval
                        if rand2 < 0.3:
                            text = self.replace_attr(text, attr, attrval)

                        # replace with neg_attr
                        elif 0.4 < rand2 < 0.5:
                            replace_val = random.choice(neg_attrvals)
                            text = text.replace(attrval, replace_val)
                        
                        # delete attr
                        elif 0.5 < rand2 < 0.6:
                            text = text.replace(attrval, '')
                        
                        # add attrval
                        elif 0.6 < rand2 < 0.7:
                            add_attrval = random.choice(neg_attrvals)
                            if given_prob(0.5):
                                if given_prob(0.5):
                                    add_attrval = add_attrval + attrval
                                else:
                                    add_attrval = attrval + add_attrval
                                text = text.replace(attrval, add_attrval)
                            else:
                                if given_prob(0.5):
                                    text = text + add_attrval
                                else:
                                    text = add_attrval + text
                
                # delete one attr no effect for match
                # elif len(attrs) >= 2 and given_prob(0.5):
                #     delete_attr = random.choice(attrs)
                #     text = text.replace(data['key_attr'][delete_attr], '')
                    # text = self.replace_attr(text, delete_attr, )

            
            # update global match rate
            if self.training:
                if match == 0:
                    self.global_match0 += 1
                else:
                    self.global_match1 += 1

            ret['match'].append(match)
            ret['feature'].append(data['feature'])
            
            text_ids = self.tokenizer.encode(text)
            ret['text_ids'].append(text_ids)
            
            ### attr match
            if data['is_fine'] or self.training:
                for k, text in data['key_attr'].items():
                    match = data['match']['图文']
                    if not data['is_fine'] and match == 0:
                        continue

                    ret['global_mask'].append(0)
                    ret['attr_mask'].append(1)

                    if self.training and match == 1 and given_prob(self.attr_match0_prob):
                        match = 0

                        # replace with random attrval of same attr
                        if given_prob(0.8):
                            cur_idx = self.attr_config['attrval_id_map'][k][text]
                            num_classes = attr_config['attr_num_classes'][k]
                            add_idx = int(random.uniform(1, num_classes))
                            text = attr_config['classid_attrval_map'][k][(cur_idx+add_idx) % num_classes]

                        # replace with neg_attrvals
                        else:
                            text = random.choice(neg_attrvals)

                    if self.training:
                        if match == 0:
                            self.attr_match0 += 1
                        else:
                            self.attr_match1 += 1

                    ret['match'].append(match)
                    ret['feature'].append(data['feature'])
                    ret['text_ids'].append(self.tokenizer.encode(text))

            return ret

    def __init__(self, raw_data_dir, tokenizer_path, length=-1, run_jieba=False,
                batch_size=128, shuffle=True, fold_idx=-1, validation_split=0.25, num_workers=1, training=True):
        self.dataset = self.__class__.InnerDataset(
            raw_data_dir=raw_data_dir, 
            tokenizer_path=tokenizer_path, 
            length=length,
            run_jieba=run_jieba
        )

        super().__init__(
            self.dataset,
            batch_size,
            shuffle,
            fold_idx,
            validation_split,
            num_workers,
            collate_fn=self.__class__.batch_collate)

