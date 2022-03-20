import re
import json


# rule = <pattern, constraint, priority>
class CausalityExtractor():
    def __init__(self):
        self.rule_num = 3
        self.limit_sent = 21
        self.center_dict = dict()
        for idx in range(1, self.rule_num + 1):
            self.center_dict[idx] = []
            with open('./center_words/' + str(idx) + '.txt', 'r', encoding='utf-8') as f:
                words = f.readlines()
                for w in words:
                    self.center_dict[idx].append(w.strip())

    def cls_center(self, center):
        for idx in range(1, self.rule_num + 1):
            if center in self.center_dict[idx]:
                return idx
        return 0

    def reduce_scope(self, text, index, mode='right', limit=0, unit='word'):
        '''
        text: original text
        index: start index
        mode: find the subsentence along the right side or left sied
        limit: limit of subsentence number, 0 is the shortest subsentece
        unit: sentence or word
        '''
        mode = 'left' if mode == 'left' else 'right'
        if unit == 'word':
            if mode == 'right':
                return index + limit
            elif mode == 'left':
                return index - limit
        elif unit == 'sentence':
            subsentences = re.split(r'[？！，；,.。]', text) # split sentence
            if subsentences[-1] == '':
                subsentences.pop()
            cur_index = -1
            cur_subsent = 0
            for i, subsent in enumerate(subsentences):
                cur_index += len(subsent) + 1 # punctuation
                if cur_index >= index:
                    cur_subsent = i
                    break

            limit_index = -1 
            if mode == 'right':
                next_subsent = min(len(subsentences) - 1, cur_subsent + limit)
                for i in range(next_subsent + 1):
                    limit_index += len(subsentences[i]) + 1
            elif mode == 'left':
                next_subsent = max(0, cur_subsent - limit)
                for i in range(next_subsent):
                    limit_index += len(subsentences[i]) + 1
                limit_index = 0 if limit_index == -1 else limit_index
            return limit_index

    '''0-默认'''
    def rule0(self, text, center_index):
        return self.rule1(text, center_index)

    '''1-由因到果-居中式'''
    def rule1(self, text, center_index,):
        idx_1 = self.reduce_scope(text, center_index[0] - 1, 'left', self.limit_sent)
        idx_2 = self.reduce_scope(text, center_index[1] + 1, 'right', self.limit_sent)
        res = dict()
        res['cause'] = text[idx_1 : center_index[0]]
        res['cause_index'] = [idx_1, center_index[0] - 1]
        res['effect'] = text[center_index[1] + 1: idx_2 + 1]
        res['effect_index'] = [center_index[1] + 1, idx_2]
        return res

    '''2-由因到果-前端式'''
    def rule2(self, text, center_index):
        idx_1 = center_index[1] + len(re.split(r'[？！，；,.。]', text[center_index[1] + 1 : ])[0])
        if idx_1 == len(text) - 1:
            idx_1 = (center_index[1] + len(text)) // 2
        idx_2 = self.reduce_scope(text, idx_1 + 1, 'right', self.limit_sent)
        res = dict()
        res['cause'] = text[center_index[1] + 1 : idx_1 + 1]
        res['cause_index'] = [center_index[1] + 1, idx_1]
        res['effect'] = text[idx_1 + 1 : idx_2 + 1]
        res['effect_index'] = [idx_1 + 1, idx_2]
        return res

    '''3-由果溯因-居中式'''
    def rule3(self, text, center_index):
        idx_1 = self.reduce_scope(text, center_index[0] - 1, 'left', self.limit_sent)
        idx_2 = self.reduce_scope(text, center_index[1] + 1, 'right', self.limit_sent)
        res = dict()
        res['cause'] = text[center_index[1] + 1 : idx_2 + 1]
        res['cause_index'] = [center_index[1] + 1, idx_2]
        res['effect'] = text[idx_1 : center_index[0]]
        res['effect_index'] = [idx_1, center_index[0] - 1]
        return res

    def extract(self, text, center, center_index, is_test=False):
        center_id = self.cls_center(center)
        if center_id == 0:
            if is_test:
                print("Can't extract any triple, just use default rule!")
            return self.rule0(text, center_index)
        elif center_id == 1:
            return self.rule1(text, center_index)
        elif center_id == 2:
            return self.rule2(text, center_index)
        elif center_id == 3:
            return self.rule3(text, center_index)