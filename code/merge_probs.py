import argparse
import json

import numpy as np
from process_data import load_pickle

# args
args = argparse.ArgumentParser(description='PyTorch Template')
args.add_argument('-s', '--save_dirs', nargs='+', default=[],
                    help='a list of save_dir of submission file to ensemble')
save_dirs = args.parse_args().save_dirs

# load data
datas = [
    load_pickle(f'{save_dir}/sub.pkl')
    for save_dir in save_dirs
]
# ensemble
merge_stats = {}
test_output = []
for i in range(len(datas[0])):
    img_name = datas[0][i]['img_name']
    match_key = datas[0][i]['match'].keys()
    match = {
        key: int(np.mean([datas[j][i]['match'][key] for j in range(len(datas))]) > 0.5)
        for key in match_key
    }
    test_output.append({
        'img_name': img_name,
        'match': match
    })
    for k, v in match.items():
        if k not in merge_stats:
            merge_stats[k] = {0: 0, 1: 0, 'total': 0}
        merge_stats[k][v] += 1
        merge_stats[k]['total'] += 1

# save_output
with open(f'../submission/result.txt', "w", encoding="utf-8") as f:
    for data in test_output:
        json_str = json.dumps(data, ensure_ascii=False)
        f.write("{}\n".format(json_str))
    f.flush()
