import torch
import os
import random
import numpy as np
import jsonlines


def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)  
	np.random.seed(seed)  # Numpy module.
	random.seed(seed)  # Python random module.
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def load_data(filename):
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append(obj)
    data = convert_index_to_bio(data)
    return data


def convert_index_to_bio(original_data):
    data = []
    for obj in original_data:
        try:
            # text sequence, label sequence, [center_start, center_end]
            d = ['', [], []]
            d[2].append(obj['center_start'])
            d[2].append(obj['center_end'])
            d[0] = obj['text']
            d[1] = ['O'] * len(obj['text'])
            for sub in obj['subject']:
                d[1][sub['start']] = 'B-SUB'
                for idx in range(sub['start'] + 1, sub['end'] + 1):
                    d[1][idx] = 'I-SUB'
            for pred in obj['predicate']:
                d[1][pred['start']] = 'B-PRED'
                for idx in range(pred['start'] + 1, pred['end'] + 1):
                    d[1][idx] = 'I-PRED'
        # cannot solve Escape character in json string
        except:
            continue
        data.append(d)
    return data


def split_sentence(raw_text, tokenizer):
    tokens = []
    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)
    return tokens


def char_level_tokenize(raw_texts, tokenizer, max_seq_len):

    tokens_list = []
    for raw_text in raw_texts:
        tokens_list.append(split_sentence(raw_text, tokenizer))

    encoding = tokenizer(tokens_list, add_special_tokens=True, is_split_into_words=True,
                         max_length=max_seq_len, padding='max_length', return_tensors='pt', truncation=True)
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    attention_mask = encoding['attention_mask']

    return input_ids, token_type_ids, attention_mask


def get_ner_metric(pred, gold):
    TP, TP_FP, TP_FN = 1e-10, 1e-10, 1e-10
    for p, g in zip(pred, gold):
        P = set(p)
        G = set(g)
        TP += len(P & G)
        TP_FP += len(P)
        TP_FN += len(G)
    precision = TP / TP_FP
    recall = TP / TP_FN
    f1  = 2 * precision * recall / (precision + recall)
    return f1, precision, recall