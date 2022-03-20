import jsonlines
from sklearn.model_selection import train_test_split

train_data = []
valid_data = []
test_data = []

shuffle = True
seed = 1020

with jsonlines.open('./data/cleaned_data.json', 'r') as reader:
    for item in reader:
        train_data.append(item)


train_data, test_data = train_test_split(train_data, test_size=0.4, shuffle=shuffle, random_state=seed)
valid_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=shuffle, random_state=seed)

print(len(train_data))
print(len(valid_data))
print(len(test_data))

with jsonlines.open('./data/financial_causality_quadruple/train.json', 'w') as writer:
    for item in train_data:
        writer.write(item)

with jsonlines.open('./data/financial_causality_quadruple/valid.json', 'w') as writer:
    for item in valid_data:
        writer.write(item)

with jsonlines.open('./data/financial_causality_quadruple/test.json', 'w') as writer:
    for item in test_data:
        writer.write(item)


