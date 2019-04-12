import json


def load_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config
