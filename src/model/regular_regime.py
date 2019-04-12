import tensorflow as tf
from src.model.base_net import RandWire
from src.model.custom_layer import RandWireLayer


class RegularRandWireNN(RandWire):
    def __init__(self, config, is_training=True):
        super(RegularRandWireNN, self).__init__(config, is_training)

        if self.graph_def is None:
            self.graph_def = [None, None, None, None]

        self.rand_wire_layer_0 = RandWireLayer(
            self.base_channel, self.base_channel, self.n // 2, self.k, self.p, self.m, 2,
            self.graph_def[0], graph_mode=config['Graph']['mode'], name='RandWire_0',
            is_training=is_training)

        self.rand_wire_layer_1 = RandWireLayer(
            self.base_channel, 2 * self.base_channel, self.n, self.k, self.p, self.m, 2,
            self.graph_def[1], graph_mode=config['Graph']['mode'], name='RandWire_1',
            is_training=is_training)

        self.rand_wire_layer_2 = RandWireLayer(
            2 * self.base_channel, 4 * self.base_channel, self.n, self.k, self.p, self.m, 2,
            self.graph_def[2], graph_mode=config['Graph']['mode'], name='RandWire_2',
            is_training=is_training)

        self.rand_wire_layer_3 = RandWireLayer(
            4 * self.base_channel, 8 * self.base_channel, self.n, self.k, self.p, self.m, 2,
            self.graph_def[2], graph_mode=config['Graph']['mode'], name='RandWire_3',
            is_training=is_training)

        if is_training:
            self.rand_wire_layer_0.save_graph(
                os.path.join(config['training_graph_path'], 'dag_0.txt'))
            self.rand_wire_layer_1.save_graph(
                os.path.join(config['training_graph_path'], 'dag_1.txt'))
            self.rand_wire_layer_2.save_graph(
                os.path.join(config['training_graph_path'], 'dag_2.txt'))
            self.rand_wire_layer_3.save_graph(
                os.path.join(config['training_graph_path'], 'dag_3.txt'))

    def __call__(self, inputs):
        with tf.variable_scope('Conv1'):
            out = tf.layers.conv2d(inputs, self.base_channel // 2, 3, 2, 'same',
                                   kernel_regularizer=tf.nn.l2_loss)
            out = tf.layers.batch_normalization(out, training=self.is_training)

        out = self.rand_wire_layer_0(out)
        out = self.rand_wire_layer_1(out)
        out = self.rand_wire_layer_2(out)
        out = self.rand_wire_layer_3(out)

        with tf.variable_scope('Classifier'):
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, 1280, 1, kernel_regularizer=tf.nn.l2_loss)
            out = tf.layers.batch_normalization(out, training=self.is_training)
            out = tf.reduce_mean(out, axis=[1, 2], keep_dims=True)
            out = tf.layers.dense(out, self.num_class, kernel_regularizer=tf.nn.l2_loss)
            out = tf.reshape(out, [-1, self.num_class])
        return out
