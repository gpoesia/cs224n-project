from base import AutoCompleteEncoder
import random

class UniformEncoder(AutoCompleteEncoder):
    'Encodes a string by removing characters uniformly at random with fixed probability.'

    def __init__(self, removal_probability=0.5):
        self.removal_probability = removal_probability

    def name(self):
        return 'UniformEncoder({:.2f})'.format(self.removal_probability)

    def encode(self, s):
        return ''.join(c for c in s if random.random() < self.removal_probability)

    def encode_batch(self, b):
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False


class FrequencyEncoder(AutoCompleteEncoder):
    'Encodes a string by removing characters uniformly at random with fixed probability.'

    def __init__(self, target_size=0.5, n_gram=2):
        """n-gram frequency-based baseline encoder. Removes most frequent n-grams until length of sequence 
        is reduced to floor(target_size * len(sequence))
        
        Keyword Arguments:
            target_size {float} -- How much of the sequence to keep when encoding (default: {0.5})
            n_gram {int} -- size of n-grams (default: {2})
        """
        self.target_size = target_size
        self.removal_probability = None

    def name(self):
        return 'UniformEncoder({:.2f})'.format(self.removal_probability)

    def encode(self, s):
        return ''.join(c for c in s if random.random() < self.removal_probability)

    def encode_batch(self, b):
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False
