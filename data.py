import json
import os
import sys
import torch

SMALL, MEDIUM, LARGE = 'small.json', 'medium.json', 'large.json'

def load_dataset(size=SMALL):
    d = os.path.dirname(__file__)
    with open(os.path.join(d, 'dataset', '{}.json'.format(size))) as f:
        return json.load(f)
