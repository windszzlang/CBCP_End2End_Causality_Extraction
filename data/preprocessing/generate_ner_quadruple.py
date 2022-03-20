import jsonlines


for type in ['train', 'valid', 'test']:
    with jsonlines.open('./data/financial_causality_quadruple/' + type + '.json', 'r') as reader:
        with jsonlines.open('./data/ner_quadruple/' + type + '.json', 'w') as writer:
            for item in reader:
                data_line = dict()
                data_line['text'] = item['text']
                data_line['cause_subject'] = []
                data_line['cause_predicate'] = []
                data_line['effect_subject'] = []
                data_line['effect_predicate'] = []
                for role in item['qas'][0]:
                    if role['question'] == '中心词':
                        data_line['center_start'] = (role['answers'][0]['start'])
                        data_line['center_end'] = (role['answers'][0]['end'])
                    elif role['question'] == '原因中的核心名词':
                        for a in role['answers']:
                            index = {'start': a['start'], 'end': a['end']}
                            data_line['cause_subject'].append(index)
                    elif role['question'] == '原因中的谓语或状态':
                        for a in role['answers']:
                            index = {'start': a['start'], 'end': a['end']}
                            data_line['cause_predicate'].append(index)
                    elif role['question'] == '结果中的核心名词':
                        for a in role['answers']:
                            index = {'start': a['start'], 'end': a['end']}
                            data_line['effect_subject'].append(index)
                    elif role['question'] == '结果中的谓语或状态':
                        for a in role['answers']:
                            index = {'start': a['start'], 'end': a['end']}
                            data_line['effect_predicate'].append(index)
                writer.write(data_line)