from predictor import Predictor

import json
import jsonlines

class Evaluator():
    def __init__(self, ckpt_path='./models/saved/best.pt'):
        self.predictor = Predictor(ckpt_path)

    # dict is unhashable and cannot be stored in set
    def _dictlist_to_strlist(self, dictlist):
        res = []
        for d in dictlist:
            d = json.dumps(d, ensure_ascii=False)
            res.append(d)
        return res

    # matric of relation pairs
    def relation_pair_matric(self, pred, true):
        right = 0.
        all = 0.
        TP = 1e-10
        TP_FP = 1e-10
        TP_FN = 1e-10
        # p, t are dict
        for p, t in zip(pred, true):
            all += 1.
            if p == t:
                right += 1.
            qas_p = p['qas'][0]
            qas_t = t['qas'][0]            
            # every question
            tmp_TP = 1
            tmp_TP_FP = 1
            tmp_TP_FN = 1
            for a, b in zip(qas_p, qas_t):
                if a['question'] == '中心词':
                    continue
                a = self._dictlist_to_strlist(a['answers'])
                b = self._dictlist_to_strlist(b['answers'])
                tmp_TP *= len(set(a) & set(b))
                tmp_TP_FP *= len(a)
                tmp_TP_FN *= len(b)
            TP += tmp_TP
            TP_FP += tmp_TP_FP
            TP_FN += tmp_TP_FN
        return {
            'accuracy': right / all,
            'f1': 2 * TP / (TP_FP + TP_FN),
            'precision': TP / TP_FP,
            'recall': TP / TP_FN
        }

    # matric of questions and answers
    def qa_matric(self, pred, true):
        right = 0.
        all = 0.
        TP_FP = 1e-10
        TP_FN = 1e-10
        TP = 1e-10
        # every data
        for p, t in zip(pred, true):
            qas_p = p['qas'][0]
            qas_t = t['qas'][0]
            # every question
            for a, b in zip(qas_p, qas_t):
                if a['question'] == '中心词':
                    continue
                all += 1.
                if a == b:
                    right += 1.
                a = self._dictlist_to_strlist(a['answers'])
                b = self._dictlist_to_strlist(b['answers'])
                TP += len(set(a) & set(b))
                TP_FP += len(a)
                TP_FN += len(b)
        return {
            'accuracy': right / all,
            'f1': 2 * TP / (TP_FP + TP_FN),
            'precision': TP / TP_FP,
            'recall': TP / TP_FN
        }

    def evaluate(self, test_dataset_path):
        pred = []
        gold = []
        with jsonlines.open(test_dataset_path) as reader:
            for obj in reader:
                output = self.predictor.predict(obj)
                pred.append(output)
                gold.append(obj)
        relation_pair_metric = self.relation_pair_matric(pred, gold)
        qa_metric = self.qa_matric(pred, gold)
        print('evaluate result:')
        print(f'****** relation_pair ******')
        # print(f'accuracy: {relation_pair_metric["accuracy"]}')
        print(f'f1: {relation_pair_metric["f1"]}')
        print(f'precision: {relation_pair_metric["precision"]}')
        print(f'recall: {relation_pair_metric["recall"]}')
        print(f'****** qa ******')
        # print(f'accuracy: {qa_metric["accuracy"]}')
        print(f'f1: {qa_metric["f1"]}')
        print(f'precision: {qa_metric["precision"]}')
        print(f'recall: {qa_metric["recall"]}')


if __name__ == '__main__':
    evaluator = Evaluator('./models/saved/CBCP_best.pt')
    evaluator.evaluate('./data/financial_causality_quadruple/valid.json')
    evaluator.evaluate('./data/financial_causality_quadruple/test.json')
    # evaluator.evaluate_efficiency('./data/unmarked_data_with_center.json')