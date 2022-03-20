from extractor import CausalityExtractor
from utils import *

import copy
from transformers import BertTokenizer

class Processor():
    def __init__(self, device, tokenizer_model='./models/pt/mixed_corpus_bert_large_model'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
        self.causality_extractor = CausalityExtractor()

    # 获取原文本
    def get_text(self, obj):
        if not isinstance(obj, dict):
            obj = json.loads(obj)
        return obj['text']

    # 获取中心词，中心词可在qas任意顺序位置但必须得存在
    def get_center(self, obj, is_test=False):
        center = dict()
        if not isinstance(obj, dict):
            obj = json.loads(obj)
        qas = obj['qas'][0]
        for qa in qas:
            if len(qa['answers']) != 0 and qa['question'] == '中心词':
                center['text'] = qa['answers'][0]['text']
                center['start'] = qa['answers'][0]['start']
                center['end'] = qa['answers'][0]['end']
                return center
        if is_test:
            print('No center word!')
        return center

    # 数据预处理，截断、分词和准备数据等操作
    def preprocess(self, content: dict, max_len=512):
        text = self.get_text(content)
        center = self.get_center(content)

        max_seq_len = min(2 + len(text), max_len)
        text = text[:max_seq_len - 2] # 截断，如果未超长不会影响

        # 分词
        input_ids, token_type_ids, attention_mask = \
            char_level_tokenize([text], self.tokenizer, max_seq_len)
        data = dict()
        data['input_ids'] = input_ids.to(self.device)
        data['token_type_ids'] = token_type_ids.to(self.device)
        data['attention_mask'] = attention_mask.to(self.device)

        # 准备中心词相关数据信息
        data['center_index'] = []
        data['center_distance'] = []

        c_start = center['start'] + 1
        c_end = center['end'] + 1
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

        data['center_index'] = torch.tensor(data['center_index']).to(self.device)
        data['center_distance'] = torch.tensor(data['center_distance']).to(self.device)
        return data

    # 从模型输出序列中抽取实体
    def extract_entity(self, model_out, content: dict):
        text = self.get_text(content)
        center = self.get_center(content)
        named_entity = ['SUB', 'PRED']
        labels = ['O']
        for e in named_entity:
            labels.append('B-' + e)
            labels.append('I-' + e)
        id2label = dict(enumerate(labels))

        # 使用因果抽离器抽出因果部分
        ce_info = self.causality_extractor.extract(text, center['text'], [center['start'], center['end']], is_test=False)
        # 使用解码器获取因果中的实体
        argument_info = self.extract_sub_pred(model_out[0], text, ce_info['cause_index'], ce_info['effect_index'], id2label)
        return argument_info

    # 序列标注解码，并根据因果句，抽离出对应论元的实体
    def extract_sub_pred(self, output_ids, text, cause_index, effect_index, id2label):
        output_ids = output_ids.cpu().int().tolist()

        cause_sub, cause_pred, effect_sub, effect_pred = dict(), dict(), dict(), dict()
        cause_sub['text'], cause_sub['end_index'] = [], []
        cause_pred['text'], cause_pred['end_index'] = [], []
        effect_sub['text'], effect_sub['end_index'] = [], []
        effect_pred['text'], effect_pred['end_index'] = [], []

        entity, last_label = '', 'O'
        for idx, (word, id) in enumerate(zip(text, output_ids)):
            this_label = id2label[id]
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
                    last_idx = idx - 1 # 上一个id
                    if flag == 'SUB':
                        if cause_index[0] <= last_idx and last_idx <= cause_index[1]:
                            cause_sub['text'].append(entity)
                            cause_sub['end_index'].append(last_idx) 
                        if effect_index[0] <= last_idx and last_idx <= effect_index[1]:
                            effect_sub['text'].append(entity)
                            effect_sub['end_index'].append(last_idx)
                    elif flag == 'PRED':
                        if cause_index[0] <= last_idx and last_idx <= cause_index[1]:
                            cause_pred['text'].append(entity)
                            cause_pred['end_index'].append(last_idx)
                        if effect_index[0] <= last_idx and last_idx <= effect_index[1]:
                            effect_pred['text'].append(entity)
                            effect_pred['end_index'].append(last_idx)
                if this_label[:1] == 'B':
                    entity = word
            last_label = this_label

        return (cause_sub, cause_pred, effect_sub, effect_pred)

    # 后处理，将抽取的数据转化为最终目标数据
    def postprocess(self, argument_info, content: dict, is_predict=False):
        info = copy.deepcopy(content)
        cause_sub, cause_pred, effect_sub, effect_pred = argument_info

        # pop and insert later to keep order
        if is_predict:
            center = info['qas'][0].pop()
        else:
            while True:
                tmp_center = info['qas'][0].pop()
                if tmp_center['question'] == '中心词':
                    center = tmp_center
                if info['qas'][0] == []:
                    break
            assert center != None, 'No center!'
            
        ## 原因中的核心名词
        frame = {
            "question": "原因中的核心名词",
            "answers": []
        }
        for text, end in zip(cause_sub['text'], cause_sub['end_index']):
            start = end - len(text) + 1
            tmp = {
                "start": start, 
                "end": end,
                "text": text,
            }
            frame['answers'].append(tmp)
        info['qas'][0].append(frame)

        ## 原因中的谓语或状态
        frame = {
            "question": "原因中的谓语或状态",
            "answers": []
        }
        for text, end in zip(cause_pred['text'], cause_pred['end_index']):
            start = end - len(text) + 1
            tmp = {
                "start": start, 
                "end": end,
                "text": text,
            }
            frame['answers'].append(tmp)
        info['qas'][0].append(frame)

        ## 中心词
        info['qas'][0].append(center)

        ## 结果中的核心名词
        frame = {
            "question": "结果中的核心名词",
            "answers": []
        }
        for text, end in zip(effect_sub['text'], effect_sub['end_index']):
            start = end - len(text) + 1
            tmp = {
                "start": start, 
                "end": end,
                "text": text,
            }
            frame['answers'].append(tmp)
        info['qas'][0].append(frame)

        ## 结果中的谓语或状态
        frame = {
            "question": "结果中的谓语或状态",
            "answers": []
        }
        for text, end in zip(effect_pred['text'], effect_pred['end_index']):
            start = end - len(text) + 1
            tmp = {
                "start": start, 
                "end": end,
                "text": text,
            }
            frame['answers'].append(tmp)
        info['qas'][0].append(frame)

        return info