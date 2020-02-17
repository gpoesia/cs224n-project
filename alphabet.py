# Convert between character and vector representations of strings.

import torch

class AlphabetEncoding:
    def encode(self, s):
        'Given a string, returns a 2D tensor representation of it.'
        raise NotImplemented()

    def decode(self, t):
        '''Given a 2D tensor where the last dimension corresponds to character embeddings,
        decodes the tensor into a string.'''
        raise NotImplemented()

class AsciiOneHotEncoding(AlphabetEncoding):
    ALPHABET_SIZE = 128

    def encode(self, s):
        b = s.encode('ascii')
        t = torch.zeros((len(s), AsciiOneHotEncoding.ALPHABET_SIZE))
        t[range(len(s)), list(b)] += 1
        return t

    def decode(self, t):
        return ''.join(map(chr, torch.argmax(t, dim=1)))

