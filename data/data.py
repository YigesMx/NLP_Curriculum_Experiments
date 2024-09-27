import json
import time

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def read_data_list(path):
    '''
    原始数据格式为 json line，每行格式如下：
    {
        'text': '浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，',
        'label': {
            'name': {'叶老桂': [[9, 11]]},
            'company': {'浙商银行': [[0, 3]]}
        }
    }
    下面将原始数据读取并返回为列表
    '''

    json_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            json_data.append(json.loads(line))
    return json_data

def data_map_process(path):

    print('start data map processing...')

    json_data = read_data_list(path)

    # 统计共有多少类别
    classes = []
    for line in json_data:
        for label_name in line['label'].keys():
            if label_name not in classes:
                classes.append(label_name)

    print('classes: ', classes)
    print('number of classes: ', len(classes))
    print()

    # 生成 tag2idx 和 idx2tag map，对每个 label_name 引入两种 tag，如B-name、I-name，并分配唯一的 idx
    tag2idx = defaultdict()
    tag2idx['O'] = 0
    idx = 1
    for label_name in classes:
        tag2idx['B-' + label_name] = idx
        idx += 1
        tag2idx['I-' + label_name] = idx
        idx += 1

    idx2tag = {v: k for k, v in tag2idx.items()}
    
    return classes, tag2idx, idx2tag


def data_process(path):
    '''
    将数据处理成 BIO 格式：
    [
        ['浙', '商', '银', '行', '企', '业', '信', '贷', '部', '叶', '老', '桂', '博', '士', '则', '从', '另', '一', '个', '角', '度', '对', '五', '道', '门', '槛', '进', '行', '了', '解', '读', '。', '叶', '老', '桂', '认', '为', '，', '对', '目', '前', '国', '内', '商', '业', '银', '行', '而', '言', '，'], 
        ['B-company', 'I-company', 'I-company', 'I-company', 'O', 'O', 'O', 'O', 'O', 'B-name', 'I-name', 'I-name', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ]
    '''
    json_data = read_data_list(path)

    processed_data = []

    for line in json_data:
        # 初始化标签为'O'
        tags = ['O'] * len(line['text'])

        for label_name in line['label']:
            for entity in line['label'][label_name]:
                for entity_range in line['label'][label_name][entity]:
                    start, end = entity_range
                    tags[start] = 'B-' + label_name
                    tags[start + 1: end + 1] = ['I-' + label_name] * (end - start)

        text_array= []
        for t in line['text']:
            text_array.append(t)

        processed_data.append([text_array, tags])

    return processed_data