import numpy as np
from scipy import sparse
def read_relation2id(fname):
    relation2id = dict()
    id2relation = dict()
    cnt = 0
    with open(fname,'r') as f:
        for line in f:
            relation = line.rstrip()
            relation2id[relation] = cnt
            id2relation[cnt] = relation
            cnt += 1
    return relation2id


def build_entity2id(fname):
    entity2id = dict()
    cnt = 0
    with open(fname,'r') as f:
        for line in f:
            h,r,ts = line.rstrip().split('\t')
            if h not in entity2id:
                entity2id[h] = len(entity2id)
            for t in ts.split():
                if t not in entity2id:
                    entity2id[t] = len(entity2id)
            if cnt % 1000 == 0:
                print('\r{}'.format(cnt),end='')
            cnt += 1
    return entity2id


def build_relation_entity_matrix(fname,relation2id,entity2id):
    # relation_head_matrix = np.zeros((len(relation2id),len(entity2id)))
    # relation_tail_matrix = np.zeros((len(relation2id),len(entity2id)))

    cnt = 0
    head_row = []
    head_col = []
    head_value = []
    tail_row = []
    tail_col = []
    tail_value = []
    with open(fname,'r') as f:
        for line in f:
            h,r,ts = line.rstrip().split('\t')
            h = entity2id[h]
            r = relation2id['.'.join(r.split('/')[1:])]

            head_row.append(r)
            head_col.append(h)
            head_value.append(1)
            for t in ts.split():
                t = entity2id[t]
                tail_row.append(r)
                tail_col.append(t)
                tail_value.append(1)
            if cnt % 1000 == 0:
                print('\r{}'.format(cnt),end='')
            cnt += 1
    num_of_relations = len(relation2id)
    num_of_entitys = len(entity2id)
    return sparse.coo_matrix((head_value,(head_row,head_col)),shape=(num_of_relations,num_of_entitys)).tocsr(),sparse.coo_matrix((tail_value,(tail_row,tail_col)),shape=(num_of_relations,num_of_entitys)).tocsr()


def build_graph(matrix_a,matrix_b):
    return matrix_a @ matrix_b.T


if __name__ == '__main__':
    import sys
    relation2id_fname = sys.argv[1]
    fb2m_fname = sys.argv[2]
    relation2id = read_relation2id(relation2id_fname)
    entity2id = build_entity2id(fb2m_fname)
    relation_head_matrix,relation_tail_matrix = build_relation_entity_matrix(fb2m_fname,relation2id,entity2id)

    import torch
    torch.save(relation_head_matrix,'relation_head_matrix.pth')
    torch.save(relation_tail_matrix,'relation_tail_matrix.pth')


