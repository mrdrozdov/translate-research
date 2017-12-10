import os
import json

from util import read_corpus, data_iter
from vocab import Vocab

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super(Model, self).__init__()

        self.embedding_size = 10
        self.hidden_dim = 10

        self.embed_src = nn.Embedding(src_vocab_size, self.embedding_size)
        self.embed_tgt = nn.Embedding(tgt_vocab_size, self.embedding_size)

        self.encoder = nn.GRU(self.embedding_size, self.hidden_dim, 1, batch_first=True)
        self.decoder = nn.GRU(self.embedding_size, self.hidden_dim, 1, batch_first=True)

        self.classify_src = nn.Linear(self.hidden_dim, src_vocab_size)
        self.classify_tgt = nn.Linear(self.hidden_dim, tgt_vocab_size)

    def forward(self, x, y):
        batch_size = x.size(0)

        xx = self.embed_src(x)
        yy = self.embed_tgt(y)

        h0 = Variable(torch.FloatTensor(1, batch_size, self.hidden_dim).fill_(0))
        encoder_out, encoder_n = self.encoder(xx, h0)
        decoder_out, _ = self.decoder(yy, encoder_n)

        logits_src = self.classify_src(
            encoder_out[:, :-1, :].contiguous().view(-1, self.hidden_dim))
        logits_tgt = self.classify_tgt(
            decoder_out[:, :-1, :].contiguous().view(-1, self.hidden_dim))

        self.encoder_loss = nn.CrossEntropyLoss()(logits_src, x[:, 1:].contiguous().view(-1))
        self.decoder_loss = nn.CrossEntropyLoss()(logits_tgt, y[:, 1:].contiguous().view(-1))
        loss = self.encoder_loss + self.decoder_loss

        return loss


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, y_hat, y):
        pass


def init_training(vocab):
    model = Model(src_vocab_size=len(vocab.src), tgt_vocab_size=len(vocab.tgt))
    optimizer = optim.Adam(model.parameters())
    nll_loss = NLLLoss()

    return model, optimizer, nll_loss


def evaluate_loss(model, data, crit):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for src_sents, tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
        src_sents_len = [len(s) for s in src_sents]

        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.data[0]
        cum_tgt_words += pred_tgt_word_num

    loss = cum_loss / cum_tgt_words
    return loss


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)

    sents_t = []
    masks = []
    for s in sents:
        pad_len = max_len - len(s)
        sents_t.append([pad_token] * pad_len + s)
        masks.append([0] * pad_len + [1] * len(s))

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def to_input_variable(sents, vocab, cuda=False, is_test=False):
    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def run(args):
    src_vocab_size = 50000
    tgt_vocab_size = 50000

    train_data_src = read_corpus(args.train_src, 'src')
    train_data_tgt = read_corpus(args.train_tgt, 'tgt')
    train_data = zip(train_data_src, train_data_tgt)

    # dev_data_src = read_corpus(args.dev_src, 'src')
    # dev_data_tgt = read_corpus(args.dev_tgt, 'tgt')
    # dev_data = zip(dev_data_src, dev_data_tgt)

    vocab = Vocab(train_data_src, train_data_tgt, src_vocab_size, tgt_vocab_size)

    model, optimizer, nll_loss = init_training(vocab)

    epoch = train_iter = 0

    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1

            src_sents_var = to_input_variable(src_sents, vocab.src)
            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt)

            loss = model(src_sents_var, tgt_sents_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if train_iter % 10 == 0:
                print("Epoch={} Train-Iter={} Loss={} Encoder-Loss={} Decoder-Loss={}".format(
                    epoch, train_iter,
                    loss.data[0], model.encoder_loss.data[0], model.decoder_loss.data[0]))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', type=str)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--train-src', default=None, type=str)
    parser.add_argument('--train-tgt', default=None, type=str)
    parser.add_argument('--dev-src', default=None, type=str)
    parser.add_argument('--dev-tgt', default=None, type=str)
    args = parser.parse_args()

    if os.path.isfile(args.config):
        args.__dict__.update(json.load(open(args.config)))

    print(args.__dict__)

    run(args)
