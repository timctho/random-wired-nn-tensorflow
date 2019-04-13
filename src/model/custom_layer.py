import tensorflow as tf
import networkx as nx
import os


class NodeOp(object):
    """Aggregate inputs and perform ReLU -> SeparableConv -> BN
    """

    def __init__(self,
                 in_degree,
                 in_channel,
                 out_channel,
                 stride,
                 is_trianing,
                 name):
        self.name = name
        self.in_degree = in_degree
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.is_training = is_trianing

    def __call__(self, inputs):
        if self.in_degree == 1:
            out = tf.squeeze(inputs, axis=-1)
        else:
            self.aggregate_w = tf.Variable(tf.zeros([self.in_degree]),
                                           name='{}_aggre_w'.format(self.name))
            out = tf.tensordot(inputs, tf.nn.sigmoid(self.aggregate_w), [[-1], [0]])

        out = tf.nn.relu(out)
        out = tf.layers.separable_conv2d(out, self.out_channel, 3, self.stride, 'same',
                                         name='{}_sep'.format(self.name),
                                         depthwise_regularizer=tf.nn.l2_loss,
                                         pointwise_regularizer=tf.nn.l2_loss)
        out = tf.layers.batch_normalization(out, training=self.is_training,
                                            name='{}_bn'.format(self.name))
        return out


class RandWireLayer(object):
    """Create Random Graph and connect as DAG graph
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 n=32,
                 k=4,
                 p=0.75,
                 m=1,
                 stride=2,
                 wire_def=None,
                 graph_mode='ws',
                 is_training=True,
                 name=None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.is_training = is_training
        self.name = name

        if wire_def is not None:
            tf.logging.info('Load graph from {}'.format(wire_def))
            self.G = nx.read_adjlist(wire_def, create_using=nx.DiGraph(), nodetype=int)
        else:
            if graph_mode == 'ws':
                self.G = nx.connected_watts_strogatz_graph(n, k, p)
            elif graph_mode == 'er':
                self.G = nx.erdos_renyi_graph(n, p, directed=True)
            elif graph_mode == 'ba':
                self.G = nx.barabasi_albert_graph(n, m)
            self.G = nx.DiGraph(self.G.edges)
        self.rev_G = self.G.reverse()

    def __call__(self, inputs):
        self.num_nodes = self.G.number_of_nodes()
        self.num_edges = self.G.number_of_edges()

        tf.logging.info(
            'Create DAG with {} nodes and {} edges'.format(self.num_nodes, self.num_edges))

        in_degree = [self.G.in_degree(i) for i in range(self.num_nodes)]
        out_tensors = [None for _ in range(self.num_nodes)] + [inputs]
        out_idx = sorted([i for i in range(self.num_nodes) if self.G.out_degree(i) == 0])
        in_idx = sorted([i for i in range(self.num_nodes) if in_degree[i] == 0])

        with tf.variable_scope(self.name):
            queue = in_idx[:]
            while queue:
                node_idx = queue.pop(0)
                in_tensors = [out_tensors[i] for i in self.rev_G.adj[node_idx]]
                if not in_tensors:
                    in_tensors = [inputs]

                with tf.variable_scope('Node_{}'.format(node_idx)):
                    cur_indeg = len(in_tensors)
                    in_tensors = tf.stack(in_tensors, axis=-1)

                    _out_c = self.out_channel if node_idx in out_idx else self.in_channel
                    _stride = self.stride if node_idx in in_idx else 1
                    out_tensors[node_idx] = NodeOp(cur_indeg,
                                                   self.in_channel,
                                                   _out_c, _stride,
                                                   self.is_training,
                                                   'Node_{}'.format(node_idx))(in_tensors)

                for i in self.G.adj[node_idx]:
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)

            with tf.variable_scope('Aggregate_out'):
                outputs = [out_tensors[i] for i in out_idx]
                outputs = tf.reduce_mean(tf.stack(outputs, axis=0), axis=0, keepdims=False)
        return outputs

    def save_graph(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        nx.write_adjlist(self.G, path)
