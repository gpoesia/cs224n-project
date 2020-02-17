from base import Encoder
import random

class UniformEncoder(Encoder):
    'Encodes a string by removing characters uniformly at random with fixed probability.'

    def __init__(self, removal_probability=0.5):
        self.removal_probability = removal_probability

    def name(self):
        return 'UniformEncoder({:.2f})'.format(self.removal_probability)

    def encode(self, s):
        return ''.join(c for c in s if random.random() < self.removal_probability)
