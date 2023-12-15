"""
This code was adapted from https://github.com/lucamasera/AWX
"""

import numpy as np
import networkx as nx
# import keras
import torch
import scipy.sparse as smat
from itertools import chain
import matplotlib.pyplot as plt


# Skip the root nodes 
to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150']

class S2VGraph(object):
    def __init__(self, g, graph_size):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.g = g
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.graph_size = graph_size

        self.max_neighbor = 0

class arff_data():
    def __init__(self, arff_file, is_GO, is_test=False, is_nan=False):
        self.X, self.Y, self.A, self.terms, self.g, g_list = parse_arff(arff_file=arff_file, is_GO=is_GO, is_test=is_test)
        if is_nan:
            indices = np.isnan(self.X).any(axis=1)
            f_indices = []
            print(indices.shape)
            max_nan = max([np.sum(np.isnan(self.X[i, :])) for i in range(len(indices))])
            for i, ind in enumerate(indices):
                if ind:
                    if np.sum(np.isnan(self.X[i, :])) >= 0.2 * max_nan:
                        f_indices.append(True)
                    else:
                        f_indices.append(False)
                else:
                    f_indices.append(False)

            if is_test:
                f_indices = np.array(f_indices)
                g_list = [g_list[i] for i in range(len(g_list)) if f_indices[i]]
                print(np.sum(f_indices), self.X.shape, self.Y.shape)
                self.X = self.X[f_indices, :]
                self.Y = self.Y[f_indices, :]
                print(self.X.shape)

        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i,j] = m[j]

        # add labels and edge_mat

        s2g = S2VGraph(self.g, len(self.terms))
        s2g.neighbors = [[] for _ in range(len(self.terms))]
        for i, j in s2g.g.edges():
            i, j = self.terms.index(i), self.terms.index(j)
            s2g.neighbors[i].append(j)
            s2g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(self.terms)):
            s2g.neighbors[i] = s2g.neighbors[i]
            degree_list.append(len(s2g.neighbors[i]))
        s2g.max_neighbor = max(degree_list)

        edges = [[self.terms.index(i), self.terms.index(j)] for (i, j) in s2g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        # deg_list = list(dict(s2g.g.degree(range(len(s2g.g)))).values())
        s2g.edge_mat = torch.LongTensor(edges).transpose(0, 1)


        self.g_list = [s2g for _ in g_list]

        # for g in g_list:
        #     # s2g = S2VGraph(g, len(self.terms))
        #     s2g = S2VGraph(self.g, len(self.terms))
        #     s2g.neighbors = [[] for _ in range(len(self.terms))]
        #     for i, j in s2g.g.edges():
        #         i, j = self.terms.index(i), self.terms.index(j)
        #         s2g.neighbors[i].append(j)
        #         s2g.neighbors[j].append(i)
        #     degree_list = []
        #     for i in range(len(self.terms)):
        #         s2g.neighbors[i] = s2g.neighbors[i]
        #         degree_list.append(len(s2g.neighbors[i]))
        #     s2g.max_neighbor = max(degree_list)
        #
        #     edges = [[self.terms.index(i), self.terms.index(j)] for (i, j) in s2g.g.edges()]
        #     edges.extend([[i, j] for j, i in edges])
        #
        #     # deg_list = list(dict(s2g.g.degree(range(len(s2g.g)))).values())
        #     s2g.edge_mat = torch.LongTensor(edges).transpose(0, 1)
        #     self.g_list.append(s2g)
        # print("# data: %d" % len(g_list))

            
def parse_arff(arff_file, is_GO=False, is_test=False):
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        g_list = []
        feature_types = []
        d = []
        cats_lens = []
        for num_line, l in enumerate(f):

            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        if is_GO:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms)==1:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i-1]))
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()
                else:
                    _, f_name, f_type = l.split()
                    
                    if f_type == 'numeric' or f_type == 'NUMERIC':
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(lambda x,i: [float(x)] if x != '?' else [np.nan])

                    else:
                        cats = f_type[1:-1].split(',')
                        cats_lens.append(len(cats))
                        d.append({key:np.eye(len(cats))[i] for i,key in enumerate(cats)})
                        feature_types.append(lambda x,i: d[i].get(x, [0.0]*cats_lens[i]))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                # print(nodes)
                y_ = np.zeros(len(nodes))
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()
                
                X.append(list(chain(*[feature_types[i](x,i) for i, x in enumerate(d_line[:len(feature_types)])])))

                # print('=====================')
                g_i = nx.DiGraph()

                for t in lab.split('@'):
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.replace('/', '.'))]] =1
                    y_[nodes_idx[t.replace('/', '.')]] = 1

                    terms = t.split('/')

                    # if len(terms) == 1:
                    #     g_i.add_edge(terms[0], 'root')
                    # else:
                    #     for i in range(2, len(terms) + 1):
                    #         g_i.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))

                g_list.append(g_i)

                # # one category
                # for t in lab.split('@'):
                #     y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, t.split('/')[0])]] =1
                #     y_[nodes_idx[t.split('/')[0]]] = 1

                # # two category
                # print('===============')
                # for t in lab.split('@'):
                #     print(t)
                #     y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, '.'.join(t.split('/')[:2]))]] = 1
                #     y_[nodes_idx['.'.join(t.split('/')[:2])]] = 1

                Y.append(y_)

        X = np.array(X)
        Y = np.stack(Y)

    # nx.draw(g)
    # plt.show()
    # print(nodes)

    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes)), nodes, g, g_list


def initialize_dataset(name, datasets):
    is_GO, train, val, test = datasets[name]
    return arff_data(train, is_GO), arff_data(val, is_GO), arff_data(test, is_GO)

def initialize_other_dataset(name, datasets):
    if 'XMC' not in name:
        is_GO, train, test = datasets[name]
        return arff_data(train, is_GO), arff_data(test, is_GO)
    else:
        g, A, X_train, y_train, X_test, y_test = initialize_XMC()
        return arff_data_XMC(g, A, X_train, y_train), arff_data_XMC(g, A, X_test, y_test)

class arff_data_XMC():
    def __init__(self, g, A, X, Y):

        # add labels and edge_mat

        self.X, self.Y, self.A, self.terms, self.g = X, Y.toarray(), A, list(range(A.shape[0])), g

        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(X))
        m = np.nanmean(X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i, j] = m[j]

        s2g = S2VGraph(g, A.shape[0])
        s2g.g = g
        self.g_list = [s2g]

def initialize_XMC():
    X_train = np.load('HMC_data/XMC/X.trn.finetune.xlnet.npy')
    X_test = np.load('HMC_data/XMC/X.tst.finetune.xlnet.npy')
    y_train = smat.load_npz('HMC_data/XMC/Y.trn.npz')
    y_test = smat.load_npz('HMC_data/XMC/Y.tst.npz')

    # print(y_train.shape)

    # print(y_train)
    # train_num, test_num = X_train.shape[0], X_test.shape[1]
    class_num = X_train.shape[1]

    g = nx.DiGraph()
    g.add_nodes_from(list(range(class_num)))
    # for row in y_train:
    #     row = [i for i, v in enumerate(row.todense().tolist()[0]) if v != 0]
    #     for x in row:
    #         for y in row:
    #             g.add_edge(x, y)
    #             g.add_edge(y, x)

    for row in y_test:
        row = [i for i, v in enumerate(row.todense().tolist()[0]) if v != 0]
        for x in row:
            for y in row:
                g.add_edge(x, y)
                g.add_edge(y, x)

    A = np.array(nx.to_numpy_matrix(g, nodelist=list(range(class_num))))
    print(type(A))


    return g, A, X_train, y_train, X_test, y_test




# initialize_XMC()



































