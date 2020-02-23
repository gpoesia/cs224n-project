# Convert between character and vector representations of strings.

import torch
from torch.nn import functional as F

class AlphabetEncoding:
    def size(self):
        'Returns the number of dimensions of the encoding.'
        raise NotImplemented()

    def encode(self, s):
        'Given a string, returns a 2D tensor representation of it.'
        raise NotImplemented()

    def decode(self, t):
        '''Given a 2D tensor where the last dimension corresponds to character embeddings,
        decodes the tensor into a string.'''
        raise NotImplemented()

    def get_padding_token(self):
        '''Returns the encoding of the padding token.'''
        raise NotImplemented()

    def get_start_token(self):
        '''Returns the encoding of the start token.'''
        raise NotImplemented()

    def get_end_token(self):
        '''Returns the encoding of the end token.'''
        raise NotImplemented()

    def encode_batch(self, b):
        '''Takes a batch of sentences as a list of strings encodes it as a tensor.
        - b: a list of strings, of length B

        Returns: a tensor of shape B x L x D, where L is the length of the longest
        sentence in the batch and D is the dimensionality of the alphabet encoding.'''
        raise NotImplemented()

    def char_to_index(self, c):
        '''Returns the index of a character in the encoding.'''
        raise NotImplemented()

    def index_to_char(self, c):
        '''Returns the character in the encoding given its index.'''
        raise NotImplemented()


class AsciiOneHotEncoding(AlphabetEncoding):
    '''One-hot encoding that only takes ASCII characters.

    Uses control ASCII characters 0, 1 and 2 for padding, start and end tokens,
    respectively.'''

    ALPHABET_SIZE = 128

    PADDING_INDEX = 0
    START_INDEX = 1
    END_INDEX = 2

    PADDING = F.one_hot(torch.scalar_tensor(0, dtype=torch.long), ALPHABET_SIZE).to(torch.float)
    START   = F.one_hot(torch.scalar_tensor(1, dtype=torch.long), ALPHABET_SIZE).to(torch.float)
    END     = F.one_hot(torch.scalar_tensor(2, dtype=torch.long), ALPHABET_SIZE).to(torch.float)

    def size(self):
        return AsciiOneHotEncoding.ALPHABET_SIZE

    def encode(self, s):
        b = s.encode('ascii')
        t = torch.zeros((len(s) + 2, AsciiOneHotEncoding.ALPHABET_SIZE))
        t[0] = AsciiOneHotEncoding.START
        t[range(1, len(s) + 1), list(b)] += 1
        t[-1] = AsciiOneHotEncoding.END
        return t

    def encode_indices(self, s):
        b = s.encode('ascii')
        return torch.tensor([AsciiOneHotEncoding.START_INDEX] +
                             list(b) +
                            [AsciiOneHotEncoding.END_INDEX], dtype=torch.long)

    def decode(self, t):
        return ''.join(map(chr, torch.argmax(t, dim=1)))

    def get_padding_token(self):
        return AsciiOneHotEncoding.PADDING

    def get_start_token(self):
        return AsciiOneHotEncoding.START

    def get_end_token(self):
        return AsciiOneHotEncoding.END

    def end_token_index(self):
        return AsciiOneHotEncoding.END_INDEX

    def padding_token_index(self):
        return AsciiOneHotEncoding.PADDING_INDEX

    def encode_batch(self, batch):
        max_length = max(map(len, batch))
        return torch.stack(
                [torch.cat([self.encode(s), self.PADDING.repeat(max_length - len(s), 1)])
                 for s in batch])

    def encode_batch_indices(self, batch):
        max_length = max(map(len, batch))
        padding_tensor = torch.tensor([AsciiOneHotEncoding.PADDING_INDEX],
                                      dtype=torch.long)
        return torch.stack(
                [torch.cat([self.encode_indices(s),
                            padding_tensor.repeat(max_length - len(s))])
                 for s in batch])

    def char_to_index(self, c):
        return ord(c)

    def index_to_char(self, i):
        return chr(i)
