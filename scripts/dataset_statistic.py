from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns


def read_dataset(fname):
    relation2freq = defaultdict(lambda:0)

    with open(fname,'r') as f:
        for line in f:
            _,_,_,relation,_ = line.split('\t')
            relation2freq[relation] += 1
    return relation2freq


def plot_relation_distribution(relation2freq,fname):
    y = []
    for rel in sorted(relation2freq,key=relation2freq.get):
        y.append(relation2freq[rel])
    plt.figure()
    plt.plot(list(range(len(relation2freq))),y)
    plt.savefig(fname)
    plt.clf()


if __name__ == '__main__':
    for i in range(10):
        relation2freq = read_dataset('fold-'+str(i)+'.train.txt')
        plot_relation_distribution(relation2freq,'fold-'+str(i)+'-train.png')



