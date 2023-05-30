import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import collections
from tcn_model import *


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, int):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


class BatchTreeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encode_dim, batch_size, use_gpu, pretrained_weight=None):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        self.bn = nn.BatchNorm1d(128)

        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        for index in range(size):
            stmt = flatten(node[index])
            stmt_embedding = self.W_c(self.embedding(Variable(self.th.LongTensor(stmt))))
            self.node_list.append(torch.max(stmt_embedding, 0)[0])

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(Variable(torch.zeros(self.batch_size, self.encode_dim)))
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return self.node_list


class BatchProgramCC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True,
                 pretrained_weight=None):
        super(BatchProgramCC, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.encoder = BatchTreeEncoder(self.vocab_size, self.embedding_dim, self.encode_dim, self.batch_size, self.gpu,
                                        pretrained_weight)
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
        self.dropout = nn.Dropout(0.2)
        self.loss_function = torch.nn.CrossEntropyLoss()
        # model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
        self.tcn = TCN(self.embedding_dim, self.label_size, [self.hidden_dim] * 2, 8, 0.2)

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def encode(self, x):

        lens = [len(item) for item in x]
        max_len = max(lens)
        encodes = []
        for i in range(len(x)):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(len(x)):
            end += lens[i]
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end
        encodes = torch.cat(seq)
        encodes = encodes.view(len(x), max_len, -1)
        encodes = torch.transpose(encodes, 1, 2)
        tcn_out = self.tcn(encodes)
        tcn_out = encodes + tcn_out
        tcn_out = F.max_pool1d(tcn_out, tcn_out.size(2)).squeeze(2)
        return tcn_out

    def forward(self, x1, x2, labels, mode='train'):
        labels = torch.LongTensor(labels).cuda()
        lvec, rvec = self.encode(x1).cuda(), self.encode(x2).cuda()
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        y = F.softmax(self.hidden2label(abs_dist), dim=1).unsqueeze(-1)
        if mode == 'train':
            loss = self.loss_function(y, labels)
            return y,loss
        elif mode == 'test':
            return y,labels

