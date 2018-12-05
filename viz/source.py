import os
from collections import defaultdict
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz


class RawGraph(object):
    r"""Raw Graph Data"""
    def __init__(self):
        r"""Initialize the class"""
        self.graph = None
        self.pr = None

    def preprocess_hero_name(self, root='data'):
        r"""Prepocess hero names

        Args
        ----
        root : str
            Root folder of raw data.

        """
        # // # allocate hero name set
        # // name_set = set([])
        # // # load names into set
        # // print('Load appear.csv')
        # // appear_df = pd.read_csv(os.path.join(root, 'appear.csv'))
        # // for i, row in appear_df.iterrows():
        # //     name_set.add(row['hero'])
        # // print('Load know.csv')
        # // know_df = pd.read_csv(os.path.join(root, 'know.csv'))
        # // for i, row in know_df.iterrows():
        # //     name_set.add(row['hero1'])
        # //     name_set.add(row['hero2'])
        # // # save name list
        # // file = open(os.path.join(root, '_name_list.p'), 'wb')
        # // pickle.dump(sorted(list(name_set)), file)
        # // file.close()

        # // # load name list
        # // file = open(os.path.join(root, '_name_list.p'), 'rb')
        # // name_list = pickle.load(file)
        # // file.close()
        # // # allocate distance dictionary
        # // dist_dict = {name: {} for name in name_list}
        # // # fill in distance symmetrically
        # // print('Compute Name Distance')
        # // step = int(len(name_list) / 25.)
        # // for i in range(len(name_list)):
        # //     name1 = name_list[i]
        # //     for j in range(i + 1, len(name_list)):
        # //         name2 = name_list[j]
        # //         dist = fuzz.ratio(name1, name2)
        # //         dist_dict[name1][name2] = dist
        # //         dist_dict[name2][name1] = dist
        # //     if (i + 1) % step == 0:
        # //         print("Compute Name Distance -- [{:>4d}/{:<4d}]".format(i + 1, len(name_list)))
        # //     else:
        # //         pass
        # // print('Compute Name Distance -- [Done]')
        # // # save distance dictionary
        # // file = open(os.path.join(root, '_name_dist.p'), 'wb')
        # // pickle.dump(dist_dict, file)
        # // file.close()

        # // # load name list
        # // file = open(os.path.join(root, '_name_list.p'), 'rb')
        # // name_list = pickle.load(file)
        # // file.close()
        # // # load distance dictionary
        # // file = open(os.path.join(root, '_name_dist.p'), 'rb')
        # // name_dist = pickle.load(file)
        # // file.close()
        # // # allocate name father
        # // print('Init Father')
        # // name_fath = {name: name for name in name_list}
        # // for i in range(len(name_list)):
        # //     for j in range(i + 1, len(name_list)):
        # //         if name_dist[name_list[i]][name_list[j]] > 90:
        # //             name_fath[name_list[j]] = name_list[i]
        # //         else:
        # //             pass
        # // print('Init Father -- [Done]')
        # // # compress in reversed order to avoid looping
        # // print('Compress Father')
        # // name_list.reverse()
        # // for name in name_fath:
        # //     while name_fath[name] != name_fath[name_fath[name]]:
        # //         name_fath[name] = name_fath[name_fath[name]]
        # // print('Compress Father -- [Done]')
        # // # truncate useless space char
        # // for name in name_fath:
        # //     name_fath[name] = name_fath[name].strip()
        # // # save name father
        # // file = open(os.path.join(root, '_name_fath.p'), 'wb')
        # // pickle.dump(name_fath, file)
        # // file.close()

        # load name father
        file = open(os.path.join(root, '_name_fath.p'), 'rb')
        self.name_fath = pickle.load(file)
        file.close()

    def from_raw(self, root='data'):
        r"""Initialize the class

        Args
        ----
        root : str
            Root folder of raw data.

        """
        # create an empty grah
        self.graph = nx.Graph()

        # read in appear-relationship
        print('Load appear.csv')
        appear_df = pd.read_csv(os.path.join(root, 'appear.csv'))
        for i, row in appear_df.iterrows():
            label1, label2 = 'hero', 'comic'
            name1, name2 = row[label1], row[label2]
            name1 = self.name_fath[name1]
            name1 = "{}_{}".format(label1, name1)
            name2 = "{}_{}".format(label2, name2)
            mode = 'appear'
            self.graph.add_edge(name1, name2, mode=mode, coop=0, cont=0)
            self.graph.nodes[name1]['label'] = label1
            self.graph.nodes[name2]['label'] = label2
            self.graph.nodes[name1]['#appear'] = 0
            self.graph.nodes[name2]['#appear'] = 0
            self.graph.nodes[name1]['#know'] = 0
            self.graph.nodes[name2]['#know'] = 0
        print('Load appear.csv -- [Done]')

        # read in know-relationship
        print('Load know.csv')
        know_df = pd.read_csv(os.path.join(root, 'know.csv'))
        for i, row in know_df.iterrows():
            label1, label2 = 'hero', 'hero'
            name1, name2 = row[label1 + '1'], row[label2 + '2']
            name1 = self.name_fath[name1]
            name2 = self.name_fath[name2]
            name1 = "{}_{}".format(label1, name1)
            name2 = "{}_{}".format(label2, name2)
            mode = 'know'
            self.graph.add_edge(name1, name2, mode=mode, coop=0, cont=0)
            self.graph.nodes[name1]['label'] = label1
            self.graph.nodes[name2]['label'] = label2
            self.graph.nodes[name1]['#appear'] = 0
            self.graph.nodes[name2]['#appear'] = 0
            self.graph.nodes[name1]['#know'] = 0
            self.graph.nodes[name2]['#know'] = 0
        print('Load know.csv -- [Done]')

        # get degree
        print('Update Degree')
        for vpair in self.graph.edges:
            for vname in vpair:
                mname = "#{}".format(self.graph.edges[vpair]['mode'])
                self.graph.nodes[vname][mname] += 1
        print('Update Degree -- [Done]')

    def pagerank(self):
        r"""Update PageRank results"""
        # get PageRank
        print('Update PageRank')
        self.pr = nx.pagerank(self.graph)
        for vname in self.graph.nodes:
            self.graph.nodes[vname]['pr'] = self.pr[vname]
            self.graph.nodes[vname]['sign'] = 0
        print('Update PageRank -- [Done]')

        # get significance
        print('Update Significance')
        for vname1, vname2 in self.graph.edges:
            for src, dst in ((vname1, vname2), (vname2, vname1)):
                self.graph.nodes[dst]['sign'] += self.pr[src]
        print('Update Significance -- [Done]')

    def weight(self):
        r"""Update Edge Weights"""
        # get cooperation
        print('Update Cooperation')
        for vname in self.graph.nodes:
            if self.graph.nodes[vname]['label'] == 'comic':
                neighbors = list(self.graph.neighbors(vname))
                for src, dst in zip(neighbors[:-1], neighbors[1:]):
                    if (src, dst) in self.graph.edges:
                        assert self.graph.nodes[src]['label'] == 'hero'
                        assert self.graph.nodes[dst]['label'] == 'hero'
                        self.graph.edges[(src, dst)]['coop'] += 1
                    else:
                        pass
            else:
                pass
        print('Update Cooperation -- [Done]')

        # get contribution
        print('Update Contribution')
        for vpair in self.graph.edges:
            if self.graph.edges[vpair]['mode'] == 'appear':
                if self.graph.nodes[vpair[0]]['label'] == 'comic':
                    vname = vpair[0]
                elif self.graph.nodes[vpair[1]]['label'] == 'comic':
                    vname = vpair[1]
                else:
                    print(vpair[0], self.graph.nodes[vpair[0]]['label'])
                    print(vpair[1], self.graph.nodes[vpair[1]]['label'])
                    raise RuntimeError()
                sign = self.graph.nodes[vname]['sign']
                deg = self.graph.nodes[vname]['#appear']
                self.graph.edges[vpair]['cont'] = sign / deg
            else:
                pass
        print('Update Contribution -- [Done]')

    def save(self, path):
        r"""Save state dictionary

        Args
        ----
        path : str
            Path to save the class data.

        """
        state_dict = {'graph': self.graph, 'pr': self.pr}
        file = open(path, 'wb')
        pickle.dump(state_dict, file)
        file.close()

    def load(self, path):
        r"""Load state dictionary

        Args
        ----
        path : str
            Path to load the class data.

        """
        file = open(path, 'rb')
        state_dict = pickle.load(file)
        file.close()
        self.graph = state_dict['graph']
        self.pr = state_dict['pr']

class ProcessedBiGraph(object):
    """Processed Majority and Minority Graph"""
    def __init__(self):
        r"""Initialize the class"""
        self.major = None
        self.minor = None

    def from_raw_graph(self, raw_graph):
        r"""Split into majority and minority subgraphs

        Args
        ----
        raw_graph : RawGraph
            Raw graph.

        """
        # split
        print('Split Majority & Minority')
        components = nx.connected_components(raw_graph.graph)
        components = sorted(components, reverse=True, key=lambda x: len(x))
        major_set = components[0]
        minor_set = set.union(*components[1:])
        self.major = raw_graph.graph.subgraph(major_set)
        self.minor = raw_graph.graph.subgraph(minor_set)
        print('Split Majority & Minority -- [Done]')

    def truncate(self, subgraph, num_heros, num_hero_comics):
        r"""Truncate a subgraph

        Args
        ----
        subgraph : networkx.Graph
            Subgraph to truncate.
        num_heros : int
            Number of hero nodes to remain.
        num_hero_comics : int
            Number of comic nodes for each remaining hero nodes to link with.
            If the value is N, then it will choose largest N and smallest N degrees from linking nodes.

        Returns
        -------
        subgraph : networkx.Graph
            Truncated subgraph.

        """
        # get hero node indices with ranking criterion
        buffer = []
        for idx in subgraph.nodes:
            node_dict = subgraph.nodes[idx]
            if node_dict['label'] == 'hero':
                buffer.append((idx, node_dict['pr']))
            else:
                pass

        # sort and truncate hero nodes
        buffer = sorted(buffer, reverse=True, key=lambda x: x[1])[0:num_heros]
        node_set = set([itr[0] for itr in buffer])

        # truncate comic neighbors of remaining hero nodes
        neigh_set = set([])
        for idx in node_set:
            neigh_list = subgraph.neighbors(idx)
            buffer = []
            for neigh in neigh_list:
                node_dict = subgraph.nodes[neigh]
                if node_dict['label'] == 'comic':
                    buffer.append((neigh, node_dict['#appear']))
                else:
                    pass
            buffer = sorted(buffer, key=lambda x: x[1])
            if num_hero_comics is not None:
                buffer = buffer[:num_hero_comics] + buffer[-num_hero_comics:]
            else:
                pass
            for itr in buffer:
                neigh_set.add(itr[0])
        return subgraph.subgraph(node_set | neigh_set)

    def layout(self, subgraph):
        r"""Update subgraph layout

        Args
        ----
        subgraph : networkx.Graph
            Subgraph to update layout.

        """
        # get layout
        print('Update Layout')
        pos = nx.kamada_kawai_layout(subgraph)
        for vname in subgraph.nodes:
            subgraph.nodes[vname]['pos'] = pos[vname]
        print('Update Layout -- [Done]')

    def save(self, path):
        r"""Save state dictionary

        Args
        ----
        path : str
            Path to save the class data.

        """
        state_dict = {'major': self.major, 'minor': self.minor}
        file = open(path, 'wb')
        pickle.dump(state_dict, file)
        file.close()

    def load(self, path):
        r"""Load state dictionary

        Args
        ----
        path : str
            Path to load the class data.

        """
        file = open(path, 'rb')
        state_dict = pickle.load(file)
        file.close()
        self.major = state_dict['major']
        self.minor = state_dict['minor']