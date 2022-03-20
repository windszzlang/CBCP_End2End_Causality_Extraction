from utils import *
from net import *
from data_generator import *
from recognizer import *

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    # filename='./record.log',
    # filemode='w',
    level=logging.INFO,
)

def train(epochs=20, print_freq=100, batch_size=16, lr=2e-5, device='cpu', max_len=512,
        data_path='./data/center_sub_pred/',
        tokenizer_path='../models/pt/chinese_roberta_wwm_ext',
        pretrained_model_path='./models/pt/chinese_roberta_wwm_ext',
        resume=False,
        load_path='./models/saved/xxx.pt',
        save_path='./models/saved/math.pt',
        multi_gpu=False):
        
    logger.info('***** prepare training *****')
    named_entity = ['SUB', 'PRED']
    labels = ['O']
    for e in named_entity:
        labels.append('B-' + e)
        labels.append('I-' + e)
    id2label = dict(enumerate(labels))
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(labels)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    if resume:
        model = torch.load(load_path)
        logger.info('loading finetuned model...')
    else:
        bert  = BertModel.from_pretrained(pretrained_model_path)
        model = ArgumentExtractorModel(bert, num_labels)
        logger.info('loading pretrained model...')
    if multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  
    model.to(device)

    logger.info('loading data...')
    train_data = load_data(data_path + 'train.json')
    valid_data = load_data(data_path + 'valid.json')
    test_data = load_data(data_path + 'test.json')

    train_data_generator = DataGenerator(train_data, id2label, label2id, device, tokenizer)

    logger.info('configuring criterion and optimizer...')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # mark best epoch
    best_dev_f1 = 0.0
    best_epoch = 0

    logger.info('***** start training *****')
    for epoch in range(1, epochs + 1):
        ## train
        logging.info(f'***** Epoch {epoch} *****')
        model.train()
        train_loss = 0.0
        cnt_train = 0
        while True:
            data = train_data_generator.get_next_batch(batch_size)
            if data == None:
                break
            # forward prop
            pred, loss = model(**data)
            pred = pred.to(device)
            if multi_gpu:
                loss = loss.sum()
            train_loss += loss.item()

            optimizer.zero_grad()
            # backward prop and update
            loss.backward()
     
            # gradient descent
            optimizer.step()

            cnt_train += 1
            if cnt_train % print_freq == 0:
                logging.info(f'Batch [{cnt_train}]: loss: {train_loss / cnt_train}')

        ## test on train set
        model.eval()
        with torch.no_grad():
            pred_entitis = []
            gold_entitis = []
            for d in train_data:
                # d = text sequence, label sequence, [center_start, center_end]
                sub_word, pred_word = recognize(d[0], d[2], tokenizer, model, id2label, device)
                pred_entitis.append(sub_word)
                pred_entitis.append(pred_word)
                sub_word, pred_word = label_sequence_to_entity(d[0], d[1], isLabel=True)
                gold_entitis.append(sub_word)
                gold_entitis.append(pred_word)
            train_f1, train_precision, train_recall = get_ner_metric(pred_entitis, gold_entitis)
            
        logging.info('***** training info *****')
        logging.info(f'Epoch [{epoch}/{epochs}]: train f1: {train_f1}, precision: {train_precision}, recall: {train_recall}')
        train_data_generator.reset()

        ## valid
        model.eval()
        with torch.no_grad():
            pred_entitis = []
            gold_entitis = []
            for d in valid_data:
                # d = text sequence, label sequence, [center_start, center_end]
                sub_word, pred_word = recognize(d[0], d[2], tokenizer, model, id2label, device)
                pred_entitis.append(sub_word)
                pred_entitis.append(pred_word)
                sub_word, pred_word = label_sequence_to_entity(d[0], d[1], isLabel=True)
                gold_entitis.append(sub_word)
                gold_entitis.append(pred_word)
            dev_f1, dev_precision, dev_recall = get_ner_metric(pred_entitis, gold_entitis)
            logging.info(f'Epoch [{epoch}/{epochs}]: dev f1: {dev_f1}, precision: {dev_precision}, recall: {dev_recall}')

        ## test
        model.eval()
        with torch.no_grad():
            pred_entitis = []
            gold_entitis = []
            for d in test_data:
                # d = text sequence, label sequence, [center_start, center_end]
                sub_word, pred_word = recognize(d[0], d[2], tokenizer, model, id2label, device)
                pred_entitis.append(sub_word)
                pred_entitis.append(pred_word)
                sub_word, pred_word = label_sequence_to_entity(d[0], d[1], isLabel=True)
                gold_entitis.append(sub_word)
                gold_entitis.append(pred_word)
            test_f1, test_precision, test_recall = get_ner_metric(pred_entitis, gold_entitis)
            logging.info(f'Epoch [{epoch}/{epochs}]: test f1: {test_f1}, precision: {test_precision}, recall: {test_recall}')

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            torch.save(model, save_path)
            # torch.save(model, save_path + '.low_version', _use_new_zipfile_serialization=False)
            logging.info('***** new score *****')
            logging.info(f'The best epoch is: {best_epoch}, with the best f1 is: {best_dev_f1}')
            logging.info('********************')

    logging.info('***** finish training *****')
    logging.info(f'The best epoch is: {best_epoch}, with the best f1 is: {best_dev_f1}')


if __name__ == '__main__':
    from utils import set_seed
    # set seed
    set_seed(667)
    train(
        epochs=20,
        print_freq=40,
        batch_size=16,
        lr=2e-5,
        device='cuda',
        max_len=512,
        data_path='./data/ner_sub_pred/',
        tokenizer_path='./models/pt/chinese_roberta_wwm_ext',
        pretrained_model_path='./models/pt/chinese_roberta_wwm_ext',
        resume=False,
        load_path='./models/saved/xxx.pt',
        save_path='./models/saved/test.pt',
        multi_gpu=False
    )
