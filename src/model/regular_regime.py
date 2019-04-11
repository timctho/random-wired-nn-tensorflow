import tensorflow as tf
from src.model.base_net import RandWire
from src.model.custom_layer import RandWireLayer


class RegularRandWireNN(RandWire):
    def __init__(self, config):
        super(RegularRandWireNN, self).__init__(config)

        if config['Graph']['dag_def']:
            graph_def = config['Graph']['dag_def']
        else:
            graph_def = [None, None, None, None]

        self.rand_wire_layer_0 = RandWireLayer(
            self.base_channel, self.base_channel, self.n // 2, self.k, self.p, self.m,
            graph_def[0], graph_mode=config['Graph']['mode'], name='RandWire_0')

        self.rand_wire_layer_1 = RandWireLayer(
            self.base_channel, 2 * self.base_channel, self.n, self.k, self.p, self.m,
            graph_def[1], graph_mode=config['Graph']['mode'], name='RandWire_1')

        self.rand_wire_layer_2 = RandWireLayer(
            2 * self.base_channel, 4 * self.base_channel, self.n, self.k, self.p, self.m,
            graph_def[2], graph_mode=config['Graph']['mode'], name='RandWire_2')

        self.rand_wire_layer_3 = RandWireLayer(
            4 * self.base_channel, 8 * self.base_channel, self.n, self.k, self.p, self.m,
            graph_def[2], graph_mode=config['Graph']['mode'], name='RandWire_3')

    def __call__(self, inputs):
        with tf.variable_scope('Conv1'):
            out = tf.layers.Conv2D(self.base_channel // 2, 3, 2, 'same')(inputs)
            out = tf.layers.BatchNormalization()(out)

        out = self.rand_wire_layer_0(out)
        out = self.rand_wire_layer_1(out)
        out = self.rand_wire_layer_2(out)
        out = self.rand_wire_layer_3(out)

        with tf.variable_scope('Classifier'):
            out = tf.nn.relu(out)
            out = tf.layers.Conv2D(1280, 1)(out)
            out = tf.layers.BatchNormalization()(out)
            out = tf.reduce_mean(out, axis=[1, 2], keep_dims=True)
            out = tf.layers.Dense(self.num_class)(out)
        return out
