import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import matplotlib
import matplotlib.pyplot as plt


def svd(M):
    u,s,vt = np.linalg.svd(M,full_matrices=False)
    V = vt.T
    S = np.diag(s)
    return np.dot(u[:,:2],np.dot(S[:2,:2],V[:,:2].T))


# def PCA(X,k=2):
#     mu = X.mean(axis=1)
#     C = np.dot(X.transpose(),X) / X.shape[0] - np.dot(mu.transpose(),mu)
#     w, v = np.linalg.eig(C)

#     idx = w.argsort()[::-1]
#     eigenVectors = v[:, idx]
#     return X @ eigenVectors[:,:k]


def plot_pca_embedding(embedding,fname,labels):
    pca = PCA(n_components=2,)
    embedding_points = pca.fit_transform(embedding)
    x = embedding_points[:,0]
    y = embedding_points[:,1]
    new_x = []
    new_y = []
    z = []
    for a,b,l in zip(x,y,labels):
        if l > 0:
            new_x.append(a)
            new_y.append(b)
            z.append(l)
    cmap = matplotlib.cm.get_cmap('viridis')
    z = np.log2(np.array(z)+1)
    normalize = matplotlib.colors.Normalize(vmin=min(z),vmax=max(z))
    colors = [cmap(normalize(value)) for value in z]
    fig,ax = plt.subplots()
    ax.scatter(new_x,new_y,color=colors,s=1)
    cax,_ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)

    plt.savefig(os.path.join(fname,'pca.png'))

    plt.clf()


def plot_tsne_embedding(embedding,fname,labels):
    tsne = TSNE(n_components=2,init='pca')
    embedding_points = tsne.fit_transform(embedding)
    x = embedding_points[:,0]
    y = embedding_points[:,1]
    new_x = []
    new_y = []
    z = []
    for a,b,l in zip(x,y,labels):
        if l > 0 :
            new_x.append(a)
            new_y.append(b)
            z.append(l)
    cmap = matplotlib.cm.get_cmap('viridis')
    z = np.log2(np.array(z)+1)
    normalize = matplotlib.colors.Normalize(vmin=min(z),vmax=max(z))
    colors = [cmap(normalize(value)) for value in z]
    fig,ax = plt.subplots()
    ax.scatter(new_x,new_y,color=colors,s=1)
    cax,_ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax,cmap=cmap,norm=normalize)
    plt.savefig(os.path.join(fname,'tsne.png'))
    plt.clf()


def plot_density(pred,gold,density,seen_idxs,fname):
    gold_relation_length = defaultdict(lambda:0)
    tp = defaultdict(lambda:0)
    for p,g in zip(pred,gold):
        gold_relation_length[g] += 1
        if p == g:
            tp[g] += 1
    x = []
    y = []
    labels = []
    for g in gold_relation_length:
        x.append(density[g])
        y.append(tp[g]*1./gold_relation_length[g])
        labels.append(1 if g in seen_idxs else 0)
    df = pd.DataFrame()
    df['density of seen nodes in neighbours'] = x
    df['precision'] = y
    df['Seen or Unseen'] = labels
    sns.scatterplot(x='density of seen nodes in neighbours',y='precision',data=df,hue='Seen or Unseen',s=10)
    plt.savefig(fname)
    plt.clf()


if __name__ == '__main__':
    embedding = np.random.rand(5000,10)
    fname = 'plt.png'
    labels = np.random.randint(0,2,size=(5000))
