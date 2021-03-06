import torch
import torch.nn as nn
import numpy as np
import pdb

from utils.module import LSTMEncoder, mean_pool, max_pool, GateNetwork, ONLSTM, max_pool2
from utils.metric import micro_precision, macro_precision
from model.GCN import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_step = 0

global_nnnn = 0  # 计算从relation_leve取了几次


class SimpleQA(nn.Module):
    def __init__(self, args):
        super(SimpleQA, self).__init__()

        if args.word_pretrained is None:
            self.word_embedding = nn.Embedding(args.n_words, args.word_dim, args.padding_idx)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(args.word_pretrained, freeze=False)

        if args.relation_pretrained is None:
            self.relation_embedding = nn.Embedding(args.n_relations, args.relation_dim, args.padding_idx)
        else:
            self.relation_embedding = nn.Embedding.from_pretrained(args.relation_pretrained, freeze=False)

        # question、relation共享的
        self.word_encoder = LSTMEncoder(
            input_size=args.word_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.question_encoder = LSTMEncoder(
            input_size=2 * args.hidden_dim,
            hidden_size=args.hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True
        )

        self.loss_fn = nn.MultiMarginLoss(margin=args.margin)
        self.model_optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

        self.ns = args.ns
        self.score_function = nn.CosineSimilarity(dim=2)

        self.all_relation_words = args.all_relation_words

        self.n_relations = args.n_relations  # 6702

        self.args = args

        global global_step
        global_step = 0

    def get_relation_embedding(self):
        # pdb.set_trace()
        all_relations = torch.tensor([i for i in range(self.n_relations)]).to(device)
        return self.relation_embedding(all_relations)

    def forward(self, question, relation, single_relation_repre):
        n_rels = relation.size()[1]  # 候选关系的数目
        question_length = (question != self.args.padding_idx).sum(dim=1).long().to(device)
        question_mask = (question != self.args.padding_idx)
        question = self.word_embedding(question)

        low_question_repre = self.word_encoder(question, question_length, need_sort=True)[0]

        high_question_repre = self.question_encoder(low_question_repre, question_length, need_sort=True)[0]
        question_repre = (low_question_repre + high_question_repre)  # bsize * seq_len * (2*hidden)

        question_repre = max_pool(question_repre, question_mask)  # bsize * (2*hidden)

        # single relation repre
        # single_relation_repre = self.get_relation_embedding()
        relation_level_relation_repre = \
            self.word_encoder(single_relation_repre.unsqueeze(1), torch.tensor([1] * self.n_relations), need_sort=True)[
                0]  # bsize * 1 * (2*hidden)

        # relation words repre
        all_relation_words = torch.tensor(self.all_relation_words).to(device)  # n_relations * max_len

        relation_words_lengths = (all_relation_words != self.args.padding_idx).sum(dim=-1).long().to(device)
        relation_words_mask = (all_relation_words != self.args.padding_idx)
        # pdb.set_trace()
        relation_words_repre = self.word_embedding(all_relation_words)

        relation_words_repre = self.word_encoder(relation_words_repre, relation_words_lengths, need_sort=True)[
            0]  # bsize * len * (2*hidden)

        # relation_repre = self.gate(single_relation_repre,relation_words_repre)
        # relation_repre = \
        #     torch.cat([relation_words_repre, relation_level_relation_repre], dim=1).max(dim=1)[0]
        relation_repre, _index = max_pool2(torch.cat([relation_level_relation_repre, relation_words_repre], dim=1),
                                           None)

        # print(_index)
        # print(_index.shape)
        # print(torch.cat([relation_level_relation_repre, relation_words_repre], dim=1).shape)
        # print("relation_level: ")
        #
        # print(torch.sum(torch.sum(_index == 0, dim=-1), dim=-1))

        relation_repre = relation_repre[relation, :]  # bsize * n_rels * hidden

        scores = self.score_function(relation_repre, question_repre.unsqueeze(1).repeat(1, n_rels, 1))

        return scores

    def train_epoch(self, train_iter):
        self.train()

        total_batch = len(train_iter)
        total = 0.
        G_losses = 0.0
        cur_batch = 1
        correct = 0
        for batch in train_iter:
            # pdb.set_trace()
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            # Traing Model, update model parameter,
            relation_repre = self.get_relation_embedding()

            # compute model loss
            scores = self.forward(question, relation, relation_repre)  # bsize * (1 + ns)
            model_loss = self.loss_fn(scores, labels)
            self.model_optimizer.zero_grad()
            model_loss.backward()
            G_losses += model_loss.item()
            self.model_optimizer.step()

            cur_batch += 1

            correct += (scores.argmax(dim=1) == labels).sum().item()
            total += bsize

            print('\r Batch {}/{}, Training Model Loss:{:.4f}, Training Acc:{:.2f}'.format(cur_batch, total_batch,
                                                                                           G_losses / cur_batch,
                                                                                           correct / total * 100),
                  end='')

    def evaluate(self, dev_iter):
        self.eval()
        total = 0
        pred = []
        gold = []
        for batch in dev_iter:
            # pdb.set_trace()
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            gold.extend(relation[range(bsize), labels].tolist())
            relation_repre = self.get_relation_embedding()
            relation_mask = (1e9 * (relation != 0).float() - 1e9)  # 1 -> 0, 0 -> -1e9
            scores = self.forward(question, relation, relation_repre)  # bsize * (1 + neg_num)

            # 所以在这里预测的全是0
            correct_idx = (scores + relation_mask).argmax(dim=1)
            pred.extend(relation[range(bsize), correct_idx].tolist())
            total += bsize

        return micro_precision(pred, gold), macro_precision(pred, gold), pred, gold

    def predict(self, dev_iter):
        self.eval()
        total = 0
        pred = []
        k_preds = []
        gold = []
        ranks = []
        for batch in dev_iter:
            question = torch.tensor(batch['question']).to(device)
            relation = torch.tensor(batch['relation']).to(device)
            labels = torch.tensor(batch['labels']).to(device)
            bsize = question.size()[0]

            gold.extend(relation[range(bsize), labels].tolist())

            relation_mask = (1e9 * (relation != 0).float() - 1e9)  # 1 -> 0, 0 -> -1e9
            scores = self.forward(question, relation)  # bsize * (1 + neg_num)
            correct_idx = (scores + relation_mask).argmax(dim=1)
            pred.extend(relation[range(bsize), correct_idx].tolist())

            k_largest_idx = (scores + relation_mask).topk(k=5, dim=1, sorted=True)[1]
            kp = np.zeros((bsize, 5))
            for i in range(bsize):
                for j in range(5):
                    kp[i][j] = relation[i][k_largest_idx[i][j]]
            k_preds.extend(kp.tolist())

            rank_idx = ((scores + relation_mask).argsort(dim=1, descending=True).argsort(dim=1)[:, 0] + 1)  # bsize * 1
            ranks.extend(rank_idx.tolist())
            total += bsize

        return gold, pred, k_preds, ranks
