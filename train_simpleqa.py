import json
import os
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataloader.simpleQA_dataloader import SimpleQADataset
from model.SimpleQA import SimpleQA
from utils.metric import cal_micro_macro_all
from utils.util import parse_args
from utils.visualize import plot_pca_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def load_relation_emb(path):
    if path.endswith("npy"):
        return np.load(path)
    context = open(path).readlines()
    context = [c.strip() for c in context]
    rel_embedding = []
    for t in context:
        data = t.split("\t")
        data = [float(x) for x in data]
        rel_embedding.append(data)
    return np.array(rel_embedding, dtype=np.float32)


def generate_folds(labelset, K=10):
    length = len(labelset)
    len_of_each_folds = length // K
    label_list = list(labelset)
    folds = []
    for i in range(0, length, len_of_each_folds):
        folds.append(label_list[i:min(i + len_of_each_folds, length)])
    return folds


def main(args):
    import time

    start_time = time.time()
    vocab = SimpleQADataset.load_vocab(args)
    vocab2, vocab_vocab2_mapping = SimpleQADataset.load_wp(args, raw_vocab=vocab)

    end_time = time.time()
    print('Loaded dataset in {:.2f}s'.format(end_time - start_time))

    args.n_words = len(vocab2.stoi)
    args.n_relations = len(vocab2.rtoi)
    # relation_word_id  [relation_num, max_relation_name]
    args.all_relation_words = vocab2.get_all_relation_words()

    if args.wup_word_pretrained_pth is not None:
        # args.word_pretrained = torch.load(args.word_pretrained_pth)
        args.word_pretrained = torch.tensor(np.load(args.wup_word_pretrained_pth).tolist(), dtype=torch.float32)
        print(' Pretrained word embedding loaded!')
    else:
        args.word_pretrained = None
        print(' Using random initialized word embedding.')

    if args.wup_relation_pretrained_pth is not None:
        # args.relation_pretrained = torch.load(args.relation_pretrained_pth)
        args.relation_pretrained = torch.tensor(load_relation_emb(args.wup_relation_pretrained_pth).tolist(),
                                                dtype=torch.float32)
        print(' Pretrained relation embedding loaded!')
    else:
        args.relation_pretrained = None
        print(' Using random initialized label word embedding.')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    base_data_dir = args.wup_data_dir
    base_save_dir = args.save_dir

    # 准确率 取平均
    all_micro = []
    all_macro = []
    seen_micro = []
    seen_macro = []
    unseen_micro = []
    unseen_macro = []

    if args.dataset == 'mix':
        print("start=%s, end=%s" % (args.start, args.end))
        for i in range(args.start, args.end + 1):
            train_fname = os.path.join(base_data_dir, 'fold-{}'.format(i) +
                                       '.train.pickle')
            dev_fname = os.path.join(base_data_dir, 'fold-{}'.format(i) +
                                     '.vaild.pickle')
            test_fname = os.path.join(base_data_dir, 'fold-{}'.format(i) +
                                      '.test.pickle')
            test_fname = os.path.join(
                '/home/user_data55/lijh/pytorch-projects/meta-sent2vec/data/SimpleQuestions/10-fold-dataset-tsv',
                'fold-{}'.format(i) + '/test.tsv')

            train_dataset, dev_dataset, test_dataset = SimpleQADataset.load_dataset(
                [
                    train_fname, dev_fname, test_fname
                ], vocab2, vocab_vocab2_mapping, args)

            args.save_dir = os.path.join(base_save_dir, 'fold-{}'.format(
                str(i)))

            train_relations = train_dataset.get_label_set()
            args.seen_idx = list(train_relations)
            args.unseen_idx = list(
                set([i for i in range(len(vocab2.rtoi))]) - set(args.seen_idx))

            label_idx = np.zeros(len(vocab2.rtoi))
            for j in range(len(vocab2.rtoi)):
                if j in train_dataset.label_freq:
                    label_idx[j] = train_dataset.label_freq[j]  # 直接用train_dataset.label_freq会有问题嘛？

            args.label_idx = label_idx
            args.writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
            if args.train:
                print(' Training Fold {}'.format(i))
                train(args, train_dataset, dev_dataset, test_dataset, vocab2,
                      SimpleQADataset.collate_fn)
            elif args.evaluate:
                with torch.no_grad():
                    print(' Test Fold {}'.format(i))
                    # All
                    test_micro_acc, test_macro_acc, pred, gold = evaluate(
                        args, test_dataset, vocab2, SimpleQADataset.collate_fn)
                    print('Test Acc :({:.2f},{:.2f})'.format(
                        test_micro_acc * 100, test_macro_acc * 100))

                    seen_macro_acc, unseen_macro_acc, all_macro1, seen_micro_acc, unseen_micro_acc, all_micro1 = cal_micro_macro_all(
                        pred,
                        gold,
                        args.seen_idx,
                        args.unseen_idx)
                    assert test_micro_acc == all_micro1, u"他们应该是相等的"
                    assert test_macro_acc == all_macro1, u"他们应该是相等的"
                    # Seen
                    # seen_micro_acc, seen_macro_acc = evaluate(
                    #     args, test_seen_dataset, vocab,
                    #     SimpleQADataset.collate_fn)

                    print('Seen Acc :({:.2f},{:.2f})'.format(
                        seen_micro_acc * 100, seen_macro_acc * 100))

                    # Unseen
                    # unseen_micro_acc, unseen_macro_acc = evaluate(
                    #     args, test_unseen_dataset, vocab,
                    #     SimpleQADataset.collate_fn)

                    print('UnSeen Acc :({:.2f},{:.2f})'.format(
                        unseen_micro_acc * 100, unseen_macro_acc * 100))

                    all_macro.append(test_macro_acc)
                    all_micro.append(test_micro_acc)
                    seen_macro.append(seen_macro_acc)
                    seen_micro.append(seen_micro_acc)
                    unseen_macro.append(unseen_macro_acc)
                    unseen_micro.append(unseen_micro_acc)

    if args.evaluate:
        print(np.mean(seen_micro), np.std(seen_micro))
        print(np.mean(seen_macro), np.std(seen_macro))
        print(np.mean(unseen_micro), np.std(unseen_micro))
        print(np.mean(unseen_macro), np.std(unseen_macro))
        print(np.mean(all_micro), np.std(all_micro))
        print(np.mean(all_macro), np.std(all_macro))


def train(args, train_dataset, dev_dataset, test_dataset, vocab, collate_fn):
    train_iter = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)
    dev_iter = DataLoader(
        dataset=dev_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)
    test_iter = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn)

    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)

    args.padding_idx = 0

    print('Building Model...', end='')
    model = SimpleQA(args).to(device)
    print('Done')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    best_acc = -1.
    patience = args.patience
    test_acc = -1.
    logfile = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    for epoch in range(args.epoch):
        model.train_epoch(train_iter)
        with torch.no_grad():
            dev_acc = model.evaluate(dev_iter)
        patience -= 1

        print(' \nEpoch {}, Patience : {}, Dev Acc : ({:.2f},{:.2f})'.format(
            epoch, patience, dev_acc[0] * 100, dev_acc[1] * 100))
        print(
            ' \nEpoch {}, Patience : {}, Dev Acc : ({:.2f},{:.2f})'.format(
                epoch, patience, dev_acc[0] * 100, dev_acc[1] * 100),
            file=logfile)

        if patience > 0 and dev_acc[0] > best_acc:
            best_acc = dev_acc[0]
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, 'model.pth'))
            print('Saving Model to {}'.format(os.path.join(args.save_dir, 'model.pth')))
            patience = args.patience

        if patience == 0:
            model = SimpleQA(args).to(device)
            model.load_state_dict(
                torch.load(os.path.join(args.save_dir, 'model.pth')))
            with torch.no_grad():
                dev_acc = model.evaluate(dev_iter)
                test_acc = model.evaluate(test_iter)
            print('Dev Acc: ({:.2f},{:.2f}), Test Acc :({:.2f},{:.2f})'.format(
                dev_acc[0] * 100, dev_acc[1] * 100, test_acc[0] * 100,
                test_acc[1] * 100))
            print(
                'Dev Acc: ({:.2f},{:.2f}), Test Acc :({:.2f},{:.2f})'.format(
                    dev_acc[0] * 100, dev_acc[1] * 100, test_acc[0] * 100,
                    test_acc[1] * 100),
                file=logfile)
            logfile.close()
            return test_acc


def evaluate(args, test_dataset, vocab, collate_fn):
    test_iter = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn)
    args.n_words = len(vocab.stoi)
    args.n_words = len(vocab.stoi)
    args.n_relations = len(vocab.rtoi)
    args.padding_idx = 0
    model = SimpleQA(args).to(device)
    print(os.path.join(args.save_dir, 'model.pth'))
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pth')))
    print("loading!!!!")
    micro_acc, macro_acc, pred, gold = model.evaluate(test_iter)
    # 计算seen unseen等
    return micro_acc, macro_acc, pred, gold


if __name__ == '__main__':
    args_parser = ArgumentParser()
    args_parser.add_argument('--config_file', '-c', default=None, type=str)
    args_parser.add_argument('--start', type=int, default=0)
    args_parser.add_argument('--end', type=int, default=9)
    args_parser.add_argument(
        '--generate',
        action="store_true",
        default=False,
    )
    args_parser.add_argument('--train', action="store_true", default=False)
    args_parser.add_argument('--evaluate', action="store_true", default=False)
    args_parser.add_argument('--visualize', action="store_true", default=False)
    args_parser.add_argument('--analysis', action="store_true", default=False)
    args_parser.add_argument('--graph_aggr', type=str, default='concat')
    args_parser.add_argument(
        '--self_loop',
        default=False,
    )
    args_parser.add_argument('--dataset', default='mix')
    args_parser.add_argument('--layer_aggr', default=None)
    args_parser.add_argument('--use_on_lstm_embedding', default=False)
    args_parser.add_argument('--chunk_size', default=10)
    args = parse_args(args_parser)
    pprint(vars(args))

    if args.train or args.evaluate or args.visualize or args.analysis:
        if args.visualize or args.analysis:
            args.fold = 2
        main(args)
