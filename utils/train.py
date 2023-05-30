import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support
from visdom import Visdom
from sklearn.utils import shuffle as reset
import torch.nn as nn
from torch.nn.parallel import DataParallel

warnings.filterwarnings('ignore')


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item['code_x'])
        x2.append(item['code_y'])
        labels.append([item['label']])
    return x1, x2, labels

def train_test_split(data, test_size=0.3, shuffle=True, random_state=2020):
    if shuffle:
        data = reset(data, random_state=random_state)
    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test


if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument('--lang')
    parser.add_argument('--lr')
    parser.add_argument('--batch')
    parser.add_argument('--gru')
    parser.add_argument('--dw')
    parser.add_argument('--epoch')
    parser.add_argument('--model_type')
    parser.add_argument('--times')
    args = parser.parse_args()
    lang = args.lang
    lr = float(args.lr)
    BATCH_SIZE = int(args.batch)
    HIDDEN_DIM = int(args.gru)
    ENCODE_DIM = int(args.dw)
    EPOCHS = int(args.epoch)
    model_type = args.model_type
    times = args.times
    data_root = '../data/'

    if model_type == 'tcn':
        from model_tcn import BatchProgramCC

        model_name = str(lang) + "_model_tcn_" + str(lr) + "_" + str(BATCH_SIZE) + "_" + str(HIDDEN_DIM) + "_" + str(
            ENCODE_DIM) + "_" + str(times)

    print("Train for %s" % model_name)

    data = pd.read_pickle(data_root + lang + '/data_all_blocks.pkl')
    train_data, test_data = train_test_split(data, test_size=0.4, shuffle=True)

    print("Data loaded.")

    word2vec = Word2Vec.load(data_root + lang + "/train/embedding/node_w2v_128").wv
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    print("Word embedding model loaded. The dimension of word vector is %s" % str(ENCODE_DIM))
    USE_GPU = True
    if lang == 'java':
        LABELS = 6
        target_names = ['0', '1', '2', '3', '4', '5']
        categories = 5
    model = BatchProgramCC(EMBEDDING_DIM, HIDDEN_DIM, MAX_TOKENS + 1, ENCODE_DIM, LABELS, BATCH_SIZE,
                           USE_GPU, embeddings)

    device_ids = [0,1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model, device_ids=device_ids).to(device)

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=lr)
    loss_function = torch.nn.CrossEntropyLoss()

    # training procedure
    print('Start training...')
    print(count_param(model))

    print(train_data)
    precision, recall, f1 = 0, 0, 0

    print('Start training...')
    for t in range(1, categories + 1):
        train_data_t = train_data[train_data['label'].isin([t, 0])]
        train_data_t.loc[train_data_t['label'] > 0, 'label'] = 1

        test_data_t = test_data[test_data['label'].isin([t, 0])]
        test_data_t.loc[test_data_t['label'] > 0, 'label'] = 1
        train_loss = []
        for epoch in range(EPOCHS):
            index = 0
            while index < len(train_data_t):
                batch = get_batch(train_data_t, index, BATCH_SIZE)
                index += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                model.zero_grad()
                output, loss = model(train1_inputs, train2_inputs, train_labels, 'train')
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        print("Testing-%d..." % t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            output, true_labels = model(test1_inputs, test2_inputs, test_labels, 'test')
            predicts.extend(output.data.cpu().numpy())
            trues.extend(true_labels)

            model.batch_size = len(test_labels)
            output = model(test1_inputs, test2_inputs)

            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)

        if lang == 'java':
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(trues, predicts, average='binary')
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(trues, predicts, average='binary')
    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))