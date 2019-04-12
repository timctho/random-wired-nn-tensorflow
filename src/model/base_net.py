import os
import tensorflow as tf


class RandWire(object):
    def __init__(self, config, is_training):
        self.base_channel = config['Model']['base_channel']
        self.is_training = is_training
        self.num_class = config['Data']['num_class']

        self.graph_def = None
        if config['Graph']['dag_def']:
            self.graph_def = config['Graph']['dag_def']
        elif config.get('training_graph_path', None) is not None:
            path = config['training_graph_path']
            if path is not None and os.path.exists(path) and len(os.listdir(path)) > 0:
                self.graph_def = sorted([os.path.join(path, p) for p in os.listdir(path)])

        self.n = self.k = self.p = self.m = 0
        mode = config['Graph']['mode']
        if mode == 'ws':
            self.n = config['Graph']['ws']['n']
            self.k = config['Graph']['ws']['k']
            self.p = config['Graph']['ws']['p']
        elif mode == 'er':
            self.n = config['Graph']['er']['n']
            self.p = config['Graph']['er']['p']
        elif mode == 'ba':
            self.n = config['Graph']['ba']['n']
            self.m = config['Graph']['ba']['m']
