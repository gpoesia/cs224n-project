from base import AutoCompleteEncoder
import random
import math
from nltk.util import ngrams
from collections import Counter
import pprint

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

    def __init__(self, dataset, compression_rate=0.5, n_gram=2):
        """n-gram frequency-based baseline encoder. Removes most frequent n-grams until length of sequence 
        is reduced to floor(target_size * len(sequence))
        
        Keyword Arguments:
            dataset {[list[strings]]} -- [all sequence examples]
            compression_rate {float} -- How much of the sequence to keep when encoding (default: {0.5})
            n_gram {int} -- size of n-grams (default: {2})
        """
        self.compression_rate = compression_rate
        self.MIN_SIZE = 6
        self.n_gram = n_gram
        # grab all n-grams in dataset and count number of each example
        self.ngram_counter = Counter(
            [l for text in dataset for l in list(zip(*[text[i:] for i in range(n_gram)]))])
    def name(self):
        return f'FrequencyEncoder({self.n_gram}-gram, target_size:{self.compression_rate})'

    @staticmethod
    def zipngram(text, n=2):
        """converts sequence to n-sized ngram tuples (no padding)
        
        Arguments:
            text {string} -- sequence to take n-grams on
        
        Keyword Arguments:
            n {int} -- size of n-grams to extract (default: {2})
        
        Returns:
            [list (tuples)] -- list of all n-grams in sequence
        """
        return list(zip(*[text[i:] for i in range(n)]))

    def encode(self, s):
        """takes a sequence and iteratively compresses the most  frequent n-gram to 
        its first and last character until the sequence is compressed to compression_rate
        
        Arguments:
            s {[string]} -- [sequence to compress]
        """

        # since strings are immutable, grab a copy
        sent = s
        # take floor b/c guarantees at least one char removed during compression
        # if math.floor(self.compression_rate*len(s)) >= self.MIN_SIZE else self.MIN_SIZE
        seq_len = math.floor(self.compression_rate*len(s))
        # get all n-grams from sequence
        seq_ngram = FrequencyEncoder.zipngram(s, self.n_gram)
        # get frequency counts of all n-grams in sequence
        ngram_counts = sorted(list(
            {ngram: self.ngram_counter[ngram] for ngram in seq_ngram}.items()), key=lambda count: -count[1])
        while len(sent) > seq_len:
            # most frequent n-gram
            freq = ''.join(ngram_counts.pop(0)[0])  #self.ngram_counter.most_common(1)[0][0])
            # get first, last characters of most frequent n-gram
            freq_comp = freq[0] + freq[-1]
            # remove first instance of n-gram from sequence
            sent = sent.split(freq, maxsplit=1)[
                0] + freq_comp + sent.split(freq, maxsplit=1)[1]
            # recalculate n-grams + frequencies in shortened sequence
            seq_ngram = FrequencyEncoder.zipngram(sent, self.n_gram)
            ngram_counts = list(
                {ngram: self.ngram_counter[ngram] for ngram in seq_ngram}.items())
            ngram_counts = sorted(
                ngram_counts, key=lambda count: count[1], reverse=True)

        return sent

    def old_encode(self, s):
        """takes a sequence and iteratively removes most frequent n-grams until
        sequence is compressed to compression_rate
        
        Arguments:
            s {[string]} -- [sequence to compress]
        
        Returns:
            [string] -- [compressed sequence]
        """
        # since strings are immutable, grab a copy
        sent = s
        # take floor b/c guarantees at least one char removed during compression 
        seq_len = math.floor(self.compression_rate*len(s)) #if math.floor(self.compression_rate*len(s)) >= self.MIN_SIZE else self.MIN_SIZE
        # get all n-grams from sequence
        seq_ngram = FrequencyEncoder.zipngram(s, self.n_gram)
        # get frequency counts of all n-grams in sequence
        
        ngram_counts = sorted(list({ngram : self.ngram_counter[ngram] for ngram in seq_ngram}.items()), key= lambda count : -count[1])
        while len(sent) > seq_len:
            # most frequent n-gram
            freq = ''.join(ngram_counts.pop(0)[0])
            # remove first instance of n-gram from sequence
            sent = sent.split(freq, maxsplit=1)[0] + sent.split(freq, maxsplit=1)[1]
            # recalculate n-grams + frequencies in shortened sequence
            print(f"after: {sent}")
            seq_ngram = FrequencyEncoder.zipngram(sent, self.n_gram)
            ngram_counts = list(
                {ngram: self.ngram_counter[ngram] for ngram in seq_ngram}.items())
            ngram_counts = sorted(
                ngram_counts, key=lambda count: count[1], reverse=True)

        return sent


    def encode_batch(self, b):
        """compress a batch of sequences
        
        Arguments:
            b {[list[string]]} -- [list of strings containing sequences to compress]
        
        Returns:
            [list[string]] -- [list of compressed sequences as strings]
        """
        return [self.encode(s) for s in b]

    def old_encode_batch(self, b):
        """compress a batch of sequences
        
        Arguments:
            b {[list[string]]} -- [list of strings containing sequences to compress]
        
        Returns:
            [list[string]] -- [list of compressed sequences as strings]
        """
        return [self.old_encode(s) for s in b]

    def is_optimizeable(self):
        return False
