import jsonlines


origin_cnt = 0
cur_cnt = 0

center_dict = dict()
for idx in range(1, 4):
    center_dict[idx] = []
    with open('./center_words/' + str(idx) + '.txt', 'r', encoding='utf-8') as f:
        words = f.readlines()
        for w in words:
            center_dict[idx].append(w.strip())

with jsonlines.open('./data/origin_data.json', 'r') as reader:
    with jsonlines.open('./data/cleaned_data.json', 'w') as writer:
        for item in reader:
            origin_cnt += 1
            # clear
            is_empty = True
            qas = item['qas'][0]
            for qa in qas:
                if len(qa['answers']) != 0 and qa['question'] == '中心词':
                    center = qa['answers'][0]['text']
                    is_empty = False
                    break
            if is_empty:
                continue
            flag = True
            for idx in range(1, 4):
                if center in center_dict[idx]:
                    flag = False
                    break
            if flag:
                continue
            cur_cnt += 1
            # clear
            del item['key'], item['status'], item['document'][0]['block_id']
            item['text'] = item['document'][0]['text']
            del item['document']
            for role in item['qas'][0]:
                for ans in role['answers']:
                    del ans['start_block'], ans['end_block'], ans['sub_answer']
            writer.write(item)

print(origin_cnt, cur_cnt)