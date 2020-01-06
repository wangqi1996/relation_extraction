import copy
import pickle

import torch
from torch.utils.data import Dataset
from collections import defaultdict
import linecache
import os
import dill
import numpy as np
import random

from utils.util import pad, load_pretrained
from dataloader.vocab import SimpleQAVocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pairwise_distances(x, y=None):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


class SimpleQADataset(Dataset):

    def __init__(self, filename, vocab, batch_size, ns=0, vocab_vocab2_mapping=None, name='train'):

        self.vocab = vocab

        self.train = name
        if self.train == 'train' or self.train == 'vaild':
            self.read_file(filename)
        elif self.train == 'test':
            self.get_test_data(filename)

        self.filename = filename
        self.batch_size = batch_size
        self.ns = ns
        subject2relation_path = 'data/subject2relation.pickle'
        self.subject_relation = pickle.load(open(subject2relation_path, "rb"))
        self.vocab_vocab2_mapping = vocab_vocab2_mapping

    def get_test_data(self, filename):

        self.label_dict = defaultdict(lambda: [])
        self.label_set = set()
        self.label_freq = defaultdict(lambda: 0)
        cnt = 0
        self.length = 0
        with open(filename, 'r') as f:
            for line in f:
                gold, neg, question = line.rstrip().split('\t')
                self.label_dict[int(gold)].append(cnt)
                self.label_set.add(int(gold))
                self.label_freq[int(gold)] += 1
                cnt += 1
                self.length += 1

    def read_file(self, filename):
        # label = relation
        self.data = pickle.load(open(filename, "rb"))

        self.label_dict = defaultdict(lambda: [])  # 每个关系包含哪些句子id
        self.label_set = set()
        self.label_freq = defaultdict(lambda: 0)  # 每个关系出现的概率
        cnt = 0  # 可以认为是句子id了吧
        self.length = 0

        for line in self.data:
            # neg应该是负采样的结果，负采样是空格分隔开
            gold = line.relation
            self.label_dict[int(gold)].append(cnt)
            self.label_set.add(int(gold))
            self.label_freq[int(gold)] += 1
            cnt += 1
            self.length += 1

    def process_line(self, item):
        # gold, neg, question = line.rstrip().split('\t')
        #
        # question = [self.vocab.stoi.get(word, 1) for word in question.split()]
        # relations = []
        # relations.append(int(gold))
        # for n in neg.split():
        #     try:
        #         idx = int(n)
        #         relations.append(idx)
        #     except ValueError:
        #         pass

        if self.train == 'train' or self.train == 'vaild':
            question = [self.vocab.stoi.get(word, 1) for word in item.question.split(' ')]
            relation = [item.relation]
            if item.subject in self.subject_relation.keys():
                cand_relation = copy.deepcopy(self.subject_relation[item.subject])
                if item.relation in cand_relation:
                    cand_relation.remove(item.relation)
                relation.extend(cand_relation)
        elif self.train == 'test':
            gold, neg, question = item.rstrip().split('\t')
            question = [self.vocab.stoi.get(word, 1) for word in question.split()]
            relation = []
            relation.append(self.vocab_vocab2_mapping[int(gold)])
            for n in neg.split():
                try:
                    idx = int(n)
                    relation.append(self.vocab_vocab2_mapping[idx])
                except ValueError:
                    print("raise valuError quesiton: %s, relation=%s" % (question, n))
                    pass
        else:
            raise ValueError(u"aaaaaaaaaaaa不支持的train类型呀, self.name=%s" % self.train)

        return {
            'question': question,
            'relations': relation,
        }

    def get_label_set(self):
        return self.label_set

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.train == 'test':
            item = linecache.getline(self.filename, index + 1)
        else:
            item = self.data[index]
        # line = linecache.getline(self.filename, item + 1)
        instance = self.process_line(item)
        if self.ns > 0:
            while len(instance['relations']) - 1 < self.ns:
                idx = random.randint(1, len(self.vocab.rtoi) - 1)
                if idx in instance['relations']:
                    continue
                instance['relations'].append(idx)
        else:

            while len(instance['relations']) - 1 < 200:
                instance['relations'].append(0)
            instance['relations'] = instance['relations'][:202]
        return instance

    @staticmethod
    def collate_fn(list_of_examples):
        """
        用来打包batch
        :param list_of_examples:
        :return:
        """
        question = np.array(pad([x['question'] for x in list_of_examples], 0))

        relation = [x['relations'] for x in list_of_examples]

        labels = [0] * len(relation)

        return {
            'question': question,
            'relation': np.array(relation),
            'labels': np.array(labels),
        }

    @staticmethod
    def load_dataset(fnames, vocab, vocab_vocab2_mapping, args):

        datasets = []
        for i, fname in enumerate(fnames):
            name = ''
            if 'train.pickle' in fname:
                name = 'train'
            if 'test.tsv' in fname:
                name = 'test'
            if 'vaild.pickle' in fname:
                name = 'vaild'
            if i == 0:
                # train
                datasets.append(SimpleQADataset(fname, vocab, args.batch_size, args.ns, vocab_vocab2_mapping, name))
            else:
                # test dev test_seen, test_unseen
                datasets.append(SimpleQADataset(fname, vocab, args.batch_size, 0, vocab_vocab2_mapping, name))

        return tuple(datasets)

    @staticmethod
    def load_vocab(args):
        """
        itor: id to relation_name
        rtoi: relation_name : id
        stoi: word: id
        relIdx2nameIdx:  {}
        relIdx2wordIdx: relation_id: [relation_name 对应的 word_id]
        """
        return torch.load(args.vocab_pth)

    @staticmethod
    def load_wp(args, raw_vocab):
        """
        返回武鹏师兄的相应配置，并返回一个映射
        :param args:
        :return:
        """
        vocab = SimpleQAVocab()
        relIdx2wordIdx_list = np.load(args.wup_relation_word_id_path)

        relIdx2wordIdx_list = [[int(i) for i in j] for j in relIdx2wordIdx_list]
        vocab.relIdx2wordIdx = {
            index: word_list for index, word_list in enumerate(relIdx2wordIdx_list)
        }

        vocab.stoi = pickle.load(open(args.wup_word_voc_path, "rb"))

        vocab.relIdx2nameIdx = {}
        vocab.itor = {}
        vocab.rtoi = {}
        relation = pickle.load(open(args.wup_rel_voc_path, 'rb'))
        for key, value in relation.items():
            if isinstance(key, int):
                vocab.itor.update({key: value})
            if isinstance(key, str):
                vocab.rtoi.update({key: value})

        vocab_vocab2_mapping = {}  # raw_id: id
        for key, value in raw_vocab.itor.items():
            new_value = vocab.rtoi.get(value, None)
            # assert new_value is not None, u"new_value 必须存在呀"
            if new_value is None:
                continue
            vocab_vocab2_mapping.update({
                key: new_value
            })
        return vocab, vocab_vocab2_mapping
