from utils import char_level_tokenize

import random
import torch
from transformers import BertTokenizer


class DataGenerator():
    def __init__(self, data, id2label, label2id, device, tokenizer, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.device = device

        self.id2label = id2label
        self.label2id = label2id

        self.total_data = dict()
        self.total_data['text'] = []
        self.total_data['center_index'] = []
        self.total_data['label'] = []
        for d in data:
            self.total_data['text'].append(d[0])
            ids = []
            for l in d[1]:
                ids.append(self.label2id[l])
            self.total_data['label'].append(ids)
            self.total_data['center_index'].append([d[2][0], d[2][1]])
        self.reset()

    def reset(self):
        self.data_ids = list(range(len(self.total_data['text'])))
        random.shuffle(self.data_ids)

    def get_next_batch(self, batch_size=64):
        data_len = len(self.data_ids)
        cur_data = []
        # cut to batch
        if data_len == 0:
            return None
        elif data_len > batch_size:
            tmp_len = batch_size
            cur_data = self.data_ids[:tmp_len]
            self.data_ids = self.data_ids[tmp_len:]
        else:
            cur_data = self.data_ids
            self.data_ids = []
        
        # initialize
        max_seq_len = self._get_max_seq_len(cur_data)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        center_index = []
        center_distance = []
        true = torch.zeros(len(cur_data), max_seq_len)

        # assign
        input_ids, token_type_ids, attention_mask = \
            char_level_tokenize([self.total_data['text'][idx] for idx in cur_data], self.tokenizer, max_seq_len)
        
        for i, idx in enumerate(cur_data):
            c_start = self.total_data['center_index'][idx][0] + 1
            c_end = self.total_data['center_index'][idx][1] + 1
            if c_end > 512:
                c_start = c_end = 0
            center_index.append([c_start, c_end])
            # center_distance
            c_dist = [511] * max_seq_len
            for j in range(max_seq_len):
                if c_start <= j <= c_end:
                    c_dist[j] = 0
                    continue
                elif j < c_start:
                    c_dist[j] = c_start - j
                else:
                    c_dist[j] = j - c_end
            center_distance.append(c_dist)

            # true
            tmp = torch.tensor(self.total_data['label'][idx])
            length = min(self.max_len - 2, len(tmp))
            # notice [CLS] token
            true[i, 1:length+1] = tmp[:length]
        
        res = dict()
        res['input_ids'] = input_ids.to(self.device)
        res['token_type_ids'] = token_type_ids.to(self.device)
        res['attention_mask'] = attention_mask.to(self.device)
        res['center_index'] = torch.tensor(center_index).to(self.device)
        res['center_distance'] = torch.tensor(center_distance).to(self.device)
        res['gold'] = true.to(self.device).float()

        return res

    def _get_max_seq_len(self, cur_data):
        res = 1
        for idx in cur_data:
            res = max(res, 2 + len(self.total_data['text'][idx]) )
        # truncate length
        return min(res, self.max_len)