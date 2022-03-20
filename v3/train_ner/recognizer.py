from utils import *

import torch
import torch.nn as nn


def recognize(text, center_index, tokenizer, model, id2label, device='cpu', max_len=512):
    '''
    :param text: a list contains many text
    :param center_index: a list contains a list of center index in a text
    '''
    max_seq_len = len(text) + 2
    while max_seq_len > max_len:
        text = text[:-1]
        max_seq_len = len(text) + 2        

    input_ids, token_type_ids, attention_mask = char_level_tokenize([text], tokenizer, max_seq_len)
    data = dict()
    data['input_ids'] = input_ids.to(device)
    data['token_type_ids'] = token_type_ids.to(device)
    data['attention_mask'] = token_type_ids.to(device)

    data['center_index'] = []
    data['center_distance'] = []

    c_start = center_index[0] + 1
    c_end = center_index[1] + 1
    if c_end > 512:
        c_start = c_end = 0
    data['center_index'].append([c_start, c_end])

    c_dist = [511] * max_seq_len
    for i in range(max_seq_len):
        if c_start <= i <= c_end:
            c_dist[i] = 0
            continue
        elif i < c_start:
            c_dist[i] = c_start - i
        else:
            c_dist[i] = i - c_end
    data['center_distance'].append(c_dist)

    data['center_index'] = torch.tensor(data['center_index']).to(device)
    data['center_distance'] = torch.tensor(data['center_distance']).to(device)
    pred = model(**data)
    cause_sub_word, cause_pred_word, effect_sub_word, effect_pred_word = label_sequence_to_entity(text, pred[0], False, id2label)
    return cause_sub_word, cause_pred_word, effect_sub_word, effect_pred_word


def label_sequence_to_entity(text, output, isLabel=False, id2label=None):
    if not isinstance(output, list):
        output = output.cpu().int().tolist()
    cause_sub_word, cause_pred_word, effect_sub_word, effect_pred_word = [], [], [], []
    entity, last_label = '', 'O'
    for idx, (word, id) in enumerate(zip(text, output)):
        if isLabel == False and id2label:
            this_label = id2label[id]
        else:
            this_label = output[idx]
        if last_label[:1] == 'O':
            if this_label[:1] == 'B':
                entity = word
            elif this_label[:1] == 'I' or this_label[:1] == 'O':
                pass
        elif last_label[:1] == 'B':
            if this_label[:1] == 'B':
                entity = word
            elif this_label[:1] == 'I':
                entity += word
            elif this_label[:1] == 'O':
                pass
        elif last_label[:1] == 'I':
            if this_label[:1] == 'I':
                entity += word
            # extract entity
            elif this_label[:1] == 'B' or this_label[:1] == 'O':
                flag = last_label[2:]
                last_idx = idx - 1
                if flag == 'CAUSESUB':
                    cause_sub_word.append(entity)
                elif flag == 'CAUSEPRED':
                    cause_pred_word.append(entity)
                elif flag == 'EFFECTSUB':
                    effect_sub_word.append(entity)
                elif flag == 'EFFECTPRED':
                    effect_pred_word.append(entity)
            if this_label[:1] == 'B':
                entity = word

        last_label = this_label
    return cause_sub_word, cause_pred_word, effect_sub_word, effect_pred_word

'''
test
'''

def recognize_test(text, center_index, tokenizer_path, model_path, id2label, device='cpu', max_len=512):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = torch.load(model_path)

    max_seq_len = len(text) + 2
    while max_seq_len > max_len:
        text = text[:-1]
        max_seq_len = len(text) + 2        

    input_ids, token_type_ids, attention_mask = char_level_tokenize([text], tokenizer, max_seq_len)
    data = dict()
    data['input_ids'] = input_ids.to(device)
    data['token_type_ids'] = token_type_ids.to(device)
    data['attention_mask'] = token_type_ids.to(device)

    data['center_index'] = []
    data['center_distance'] = []

    c_start = center_index[0] + 1
    c_end = center_index[1] + 1
    if c_end > 512:
        c_start = c_end = 0
    data['center_index'].append([c_start, c_end])

    c_dist = [511] * max_seq_len
    for i in range(max_seq_len):
        if c_start <= i <= c_end:
            c_dist[i] = 0
            continue
        elif i < c_start:
            c_dist[i] = c_start - i
        else:
            c_dist[i] = i - c_end
    data['center_distance'].append(c_dist)

    data['center_index'] = torch.tensor(data['center_index']).to(device)
    data['center_distance'] = torch.tensor(data['center_distance']).to(device)
    pred = model(**data)
    sub_word, pred_word = label_sequence_to_entity(text, pred[0], False, id2label)
    print('sub word:', sub_word)
    print('pred word:', pred_word)

if __name__ == '__main__':
    named_entity = ['SUB', 'PRED']
    labels = ['O']
    for e in named_entity:
        labels.append('B-' + e)
        labels.append('I-' + e)
    id2label = dict(enumerate(labels))
    text = ['中国6月外汇储备意外上升，主要得益于储备资产价格上涨带来的正向估值效应。']
    center_index = [[15, 17]]
    recognize_test(text, center_index,
        tokenizer_path='./models/pt/chinese_roberta_wwm_ext',
        model_path='./models/saved/baseline_0327.pt',
        id2label=id2label,
        device='cuda'
    )
