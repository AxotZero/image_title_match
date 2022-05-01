import yaml
import json
import itertools
import pickle
from pdb import set_trace as bp

from tqdm import tqdm
from transformers import BertTokenizer

from constant import PADDING, UNK


def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict


def load_yaml(fp):
    return yaml.load(open(fp, "r", encoding='utf-8'),
                     Loader=yaml.SafeLoader)


def save_yaml(data, fp):
    yaml.dump(
        data,
        open(fp, "w", encoding="utf-8"),
        allow_unicode=True,
        default_flow_style=False)


def load_raw_data(path, length=-1):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, l in enumerate(f.readlines()):
            if length != -1 and i >= length:
                break
            data.append(json.loads(l))
    return data


def stat_key_attr(data):
    """
    generate something like
    {
        '版型': {
            'num': 195,
            'num_classes': 3,
            'type': {
                '修身型': 55, 
                '宽松型': 65, 
                '标准型': 75}},
        '穿着方式': {
            'num': 73, 
            'num_classes': 2, 
            'type': {
                '套头': 59, 
                '开衫': 14}},
    """

    attr_stat = {}
    for d in data:
        attrs = d['key_attr']
        for k, v in attrs.items():
            if k not in attr_stat:
                attr_stat[k] = {
                    'num': 0,
                    'num_classes': 0,
                    'type': {}
                }
            if v not in attr_stat[k]['type']:
                attr_stat[k]['num_classes'] += 1
                attr_stat[k]['type'][v] = 0
            attr_stat[k]['num'] += 1
            attr_stat[k]['type'][v] += 1
    return attr_stat


def save_json(data, fp):
    with open(fp, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, indent=4)
        f.write("{}\n".format(json_str))
        f.flush()


def load_json(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


def save_json_file(list_data, fp):
    with open(fp, "w", encoding="utf-8") as f:
        for data in list_data:
            json_str = json.dumps(data, ensure_ascii=False)
            f.write("{}\n".format(json_str))
        f.flush()


def make_vocabs(fine_raw, coarse_raw, char_level=True, save_dir='./preprocess/v1'):
    texts = [d['title'].lower() for d in coarse_raw + fine_raw]
    vocabs = list(set(itertools.chain.from_iterable(texts)))

    vocabs = [PADDING, UNK] + vocabs
    save_path = f'{save_dir}/vocab.txt'
    with open(save_path, mode='w+', encoding='utf-8') as f:
        for word in vocabs:
            f.write(word + '\n')
    return save_path


def stat_class_count(data, attr_config):
    attr_class_count = {
        attr: {'total': 0, 'class_count': [
            0]*num_classes, 'num_classes': num_classes}
        for attr, num_classes in attr_config['attr_num_classes'].items()
    }
    for d in data:
        attrs = d['key_attr']
        for k, v in attrs.items():
            attr_class_count[k]['total'] += 1
            attr_class_count[k]['class_count'][attr_config['attrval_id_map'][k][v]] += 1
    # for k, v in attr_class_count.items():
    #     v['class_weight'] =  [v['total'] / (v['num_classes'] * count) for count in v['class_count']]
    return attr_class_count


def load_attr(path):
    data = json.load(open(path, 'r'))

    attr_id_map = {attr: i for i, attr in enumerate(data)}

    id_attr_map = {v: k for k, v in attr_id_map.items()}


    attr_num_classes = {}
    attr_vals = []
    attrval_replace_map = {}
    attrval_id_map = {}
    for attr, attrvals in data.items():
        d = {}
        for class_id, vals in enumerate(attrvals):
            vals = vals.split('=')
            replace_val = vals[0]
            attr_vals.append(replace_val)
            d[replace_val] = class_id
            for idx, val in enumerate(vals):
                attrval_replace_map[val] = replace_val
        attrval_id_map[attr] = d
        attr_num_classes[attr] = class_id + 1

    attrval_attr_map = {
        val: k for k, attrvals in attrval_id_map.items()
        for val in attrvals
    }

    classid_attrval_map = {}
    for attr, id_map in attrval_id_map.items():
        classid_attrval_map[attr] = [attrval for attrval in id_map]


    attr_config = {
        'attr_id_map': attr_id_map,
        'id_attr_map': id_attr_map,
        'attr_num_classes': attr_num_classes,
        'attrvals': attr_vals,
        'attrval_id_map': attrval_id_map,
        'attrval_attr_map': attrval_attr_map,
        'attrval_replace_map': attrval_replace_map,
        'classid_attrval_map': classid_attrval_map
    }
    return attr_config


def load_tokenizer(vocab_path, attr_config):
    tokenizer = BertTokenizer(vocab_path)
    # tokenizer.add_special_tokens(dict(additional_special_tokens=attr_config['attrvals']))
    # tokenizer.
    return tokenizer


def preprocess_train_data(data_path, attr_config, is_fine=False):
    """
    add following (key: value) to each element of data,
        'title_ids': [74, 3, 184, 179, 13, 71, 465], 
        'is_coarse': False, 
        'global_match': 1, 
        'attr_match': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        'attr_type_label': [[0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0], [0, 0, 1]]}
    """
    attrvals = attr_config['attrvals']
    attrval_attr_map = attr_config['attrval_attr_map']
    attrval_replace_map = attr_config['attrval_replace_map']

    for d in tqdm(data):
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
            d['key_attr'] = {}
            for val in attrvals:
                if val in d['title']:
                    d['key_attr'][attrval_attr_map[val]] = val

    return data
    # save_pickle(data, f'{save_dir}/{save_name}')


def preprocess_test_data(save_dir, data_path, attr_config, stage='A'):
    # tokenizer = load_tokenizer(f'{save_dir}/vocab.txt', attr_config)

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
    save_pickle(data, f'{save_dir}/test{stage}.pkl')


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--coarse_data_path', default=None, type=str,
                      help='path to raw coarse data')
    args.add_argument('--fine_data_path', default=None, type=str,
                      help='path to raw fine data')
    args.add_argument('--attr_to_attrvals_path', default=None, type=str,
                      help='path to raw attr_to_attrvals.json')
    args.add_argument('--save_dir', default=None, type=str,
                      help='path to directory to save preprocess data')

    args = args.parse_args()



    SAVE_DIR = args.save_dir

    COARSE_PATH = args.coarse_data_path
    FINE_PATH = args.fine_data_path
    ATTR_PATH = args.attr_to_attrvals_path
    LENGTH = -1

    # load attr config
    print('load_attr')
    attr_config = load_attr(ATTR_PATH)
    save_json(attr_config, f'{SAVE_DIR}/attr_config.json')

    # load data
    fine_raw = load_raw_data(FINE_PATH, LENGTH)
    coarse_raw = load_raw_data(COARSE_PATH, LENGTH)


    print('process fine')
    preprocess_train_data(SAVE_DIR, 'fine.pkl', fine_raw, attr_config, True)

    print('process coarse')
    preprocess_train_data(SAVE_DIR, 'coarse.pkl', coarse_raw, attr_config, False)

    # print('process test')
    # attr_config = load_json(f'{SAVE_DIR}/attr_config.json')
    # preprocess_test_data(SAVE_DIR, TEST_PATH, attr_config, STAGE)
