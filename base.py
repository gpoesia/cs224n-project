class Encoder:
    '''Base class for an auto-complete scheme encoder: given a string, an encoder
    able to return a shortened version of it that can be used to reconstruct the
    original string by a trained decoder.'''

    def name(self):
        raise NotImplemented()

    def encode(self, s):
        'Returns a shortened version of the string s.'
        raise NotImplemented()
