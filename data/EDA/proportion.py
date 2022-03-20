import jsonlines


center_dict = dict()
for idx in range(1, 4):
    center_dict[idx] = []
    with open('./center_words/' + str(idx) + '.txt', 'r', encoding='utf-8') as f:
        words = f.readlines()
        for w in words:
            center_dict[idx].append(w.strip())

counter = [0, 0, 0, 0]
empty = 0

with jsonlines.open('./data/cleaned_data.json') as reader:
    for item in reader:
        qas = item['qas'][0]
        is_empty = True
        for qa in qas:
            if len(qa['answers']) != 0 and qa['question'] == '中心词':
                center = qa['answers'][0]['text']
                is_empty = False
        if is_empty:
            empty += 1
            continue
        flag = True
        for idx in range(1, 4):
            if center in center_dict[idx]:
                counter[idx] += 1
                flag = False
                break
        if flag:
            print(center)
            counter[0] += 1

print(counter, empty)
        
