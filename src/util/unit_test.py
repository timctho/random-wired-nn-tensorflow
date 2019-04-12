import unittest
import shutil
import os
import sys

sys.path.append('../../')

import tensorflow as tf
import networkx as nx
from src.model.small_regime import SmallRandWireNN
from src.util.misc import load_config


class GraphTest(unittest.TestCase):

    def test_load_graph(self):
        graph_path = 'tmp/rand_graph/dag_0.txt'
        G = nx.read_adjlist(graph_path, create_using=nx.DiGraph(), nodetype=int)
        print(G.edges)
        print(G.number_of_nodes())
        print(type(G.in_degree))
        l = list(G.in_degree)
        print(l[0])
        for i in range(G.number_of_nodes()):
            print('In-deg of node {} = {}'.format(i, G.in_degree(i)))


class ModelTest(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.config = load_config('../../src/config/default.json')
        self.ph = tf.placeholder(tf.float32, [1, 224, 224, 3])

    def test_create_model_default(self):
        model = SmallRandWireNN(self.config, False)
        out = model(self.ph)

    def test_create_model_training(self):
        shutil.rmtree('tmp')
        self.config['training_graph_path'] = os.path.join('tmp/rand_graph')
        model = SmallRandWireNN(self.config, True)
        out = model(self.ph)

    def test_create_model_reload(self):
        shutil.rmtree('tmp')
        self.config['training_graph_path'] = os.path.join('tmp/rand_graph')
        print('Create first graph')
        model = SmallRandWireNN(self.config, True)
        out = model(self.ph)

        print('Create second graph')
        tf.reset_default_graph()
        self.ph = tf.placeholder(tf.float32, [1, 224, 224, 3])
        del model
        model = SmallRandWireNN(self.config, True)
        out = model(self.ph)


if __name__ == '__main__':
    unittest.main()
