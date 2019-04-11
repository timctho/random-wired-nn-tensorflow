import tensorflow as tf


class RandWire(object):
    def __init__(self, config):
        self.base_channel = config['Model']['base_channel']
        self.num_class = config['Data']['num_class']

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
