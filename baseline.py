from base import AutoCompleteEncoder

import random
import math
from nltk.util import ngrams
from collections import Counter
import pprint
import numpy
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

class CurriculumUniformEncoder(AutoCompleteEncoder):
    'Encodes a string by removing characters uniformly at random with fixed probability.'

    def __init__(self, removal_probability=0.5, curriculum_probability=0.9):
        self.removal_probability = removal_probability
        self.curriculum_probability = curriculum_probability

    def name(self):
        return 'CurriculumUniformEncoder({:.2f})'.format(self.removal_probability)

    def encode(self, s):
        return ''.join(c for c in s if random.random() < self.removal_probability)

    def encode_batch(self, b):
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False

    def curriculum_step(self):
        self.removal_probability *= self.curriculum_probability


class UniformEncoderConstantDrop(AutoCompleteEncoder):
    'Encodes a string by removing num_char characters uniformly.'

    def __init__(self, num_chars=3):
        self.num_chars = num_chars

    def name(self):
        return 'UniformEncoderConstantDrop({:.2f})'.format(self.num_chars)

    def encode(self, s):
        # remove random indices of
        seq = s
        assert self.num_chars < len(s), "number of chars to remove is larger than number of chars in sequence!"
        inds = []
        for _ in range(self.num_chars):
            # grab a random index to remove
            ind = random.randrange(len(seq))
            inds.append(ind)
            seq = seq[:ind] + seq[ind+1:]
        return seq

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
        assert n_gram > 2, "can't have an n-gram of size 2 with model that keeps first and last n-gram!"
        # grab all n-grams in dataset and count number of each example
        self.ngram_counter = Counter(
            [l for text in dataset for l in list(zip(*[text[i:] for i in range(n_gram)]))])
    def name(self):
        return (
                'FrequencyEncoder({}-gram, target_size:{})'
                .format(self.n_gram, self.compression_rate))

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

    def encode_batch(self, b):
        """compress a batch of sequences

        Arguments:
            b {[list[string]]} -- [list of strings containing sequences to compress]

        Returns:
            [list[string]] -- [list of compressed sequences as strings]
        """
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False

def normalize(lst): return [x/sum(lst) for x in lst]
def flatten(lst): return [i for l in lst for i in l]
def zipngram(text, n=2):
    return list(map(''.join, zip(*[text[i:] for i in range(n)])))


class NonDeterministicFrequencyEncoder(AutoCompleteEncoder):
    'Encodes a string by removing n-grams according to their occurence distribution'

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
        assert n_gram > 2, "can't have an n-gram of size 2 with model that keeps first and last n-gram!"
        # grab all n-grams in dataset and count number of each example
        self.ngram_counter = Counter(flatten(list(map(lambda text: zipngram(text, n=n_gram), dataset))))

    def name(self):
        return ('NonDeterministicFrequencyEncoder({}-gram, target_size:{})'
                .format(self.n_gram, self.compression_rate))

    def encode(self, s):
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
        target_seq_len = math.floor(self.compression_rate*len(sent)) #if math.floor(self.compression_rate*len(s)) >= self.MIN_SIZE else self.MIN_SIZE
        # get all n-grams from sequence
        seq_ngram = zipngram(sent, self.n_gram)
        # get frequency counts of all n-grams in sequence
        ngram_probs = normalize(list(map(lambda x: self.ngram_counter[x], seq_ngram)))

        while len(sent) > target_seq_len:
            # sampled n-gram
            n_gram = numpy.random.choice(seq_ngram, p=ngram_probs)
            # remove first instance of n-gram from sequence
            sent = sent.replace(n_gram,"",1)
            # recalculate n-grams + frequencies in shortened sequence
            seq_ngram = zipngram(sent, self.n_gram)
            counts = list(map(lambda x: self.ngram_counter[x], seq_ngram))
            if sum(counts) == 0:
                ngram_probs = [1.0/len(seq_ngram)]*len(seq_ngram)
            else:
                ngram_probs = normalize(counts)
        return sent

    def encode_batch(self, b):
        """compress a batch of sequences
        Arguments:
            b {[list[string]]} -- [list of strings containing sequences to compress]
        Returns:
            [list[string]] -- [list of compressed sequences as strings]
        """
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False

class FuzzyFrequencyEncoder(AutoCompleteEncoder):
    'Encodes a string by removing n-grams according to their occurence distribution'

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
        assert n_gram > 2, "can't have an n-gram of size 2 with model that keeps first and last n-gram!"
        # grab all n-grams in dataset and count number of each example
        self.ngram_counter = Counter(flatten(list(map(lambda text: zipngram(text, n=n_gram), dataset))))

    def name(self):
        return ('FuzzyFrequencyEncoder({}-gram, target_size:{})'
                .format(self.n_gram, self.compression_rate))

    def encode(self, s):
        """takes a sequence and iteratively removes most frequent n-grams until
        sequence is compressed to compression_rate

        Arguments:
            s {[string]} -- [sequence to compress]

        Returns:
            [string] -- [compressed sequence]
        """
        sent = s
        target_seq_len = math.floor(self.compression_rate*len(sent))
        while len(sent) > target_seq_len:
            seq_ngram = zipngram(sent, self.n_gram)
            most_common = max(seq_ngram, key=lambda x: self.ngram_counter[x])
            if random.random() > 0.5:
                sent = sent.replace(most_common,"",1)
            else:
                remove_ind = random.randint(0,len(sent)-1)
                sent = sent[:remove_ind]+sent[remove_ind+1:]
        return sent

    def encode_batch(self, b):
        """compress a batch of sequences
        Arguments:
            b {[list[string]]} -- [list of strings containing sequences to compress]
        Returns:
            [list[string]] -- [list of compressed sequences as strings]
        """
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False





class FrequencyEncoderConstantDrop(AutoCompleteEncoder):
    'Encodes a string by removing characters uniformly at random with fixed probability.'

    def __init__(self, dataset, num_chars=3, n_gram=2):
        """n-gram frequency-based baseline encoder. Removes most frequent n-grams until length of sequence
        is reduced to ceil(num_chars / n_gram) * n_gram

        Keyword Arguments:
            dataset {[list[strings]]} -- [all sequence examples]
            compression_rate {float} -- How much of the sequence to keep when encoding (default: {0.5})
            n_gram {int} -- size of n-grams (default: {2})
        """
        self.num_num_chars = num_chars
        self.num_chars = num_chars #math.ceil(num_chars / n_gram) * n_gram
        self.num_chars_low = math.floor(
            num_chars / n_gram) * n_gram
        self.n_gram = n_gram
        assert n_gram > 2, "can't have an n-gram of size 2 with model that keeps first and last n-gram!"
        # grab all n-grams in dataset and count number of each example
        assert min(len(i) for i in dataset) > n_gram, "dataset has too-small sequences!"
        self.ngram_counter = Counter(
            [l for text in dataset for l in list(zip(*[text[i:] for i in range(n_gram)]))])

    def name(self):
        return ('FrequencyEncoderConstantDrop({}-gram, drop:{})'
                .format(self.n_gram, self.num_chars))

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
        seq_len = len(s) - self.num_chars if (len(s) - self.num_chars) > self.n_gram else self.num_chars
        assert len(s) > self.num_chars, "number of chars to remove is larger than size of sequence! either filter out short sequences or reduce n-gram size."
        # get all n-grams from sequence
        seq_ngram = FrequencyEncoder.zipngram(s, self.n_gram)
        # get frequency counts of all n-grams in sequence
        ngram_counts = sorted(list(
            {ngram: self.ngram_counter[ngram] for ngram in seq_ngram}.items()), key=lambda count: -count[1])
        while len(sent) > seq_len:
#             import pdb; pdb.set_trace()
            freq = ''.join(ngram_counts.pop(0)[0])
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
#         print(len(s), self.num_num_chars, len(sent), self.num_chars)
        return sent

    def encode_batch(self, b):
        """compress a batch of sequences

        Arguments:
            b {[list[string]]} -- [list of strings containing sequences to compress]

        Returns:
            [list[string]] -- [list of compressed sequences as strings]
        """
        assert min(len(i) for i in b) > self.n_gram, "dataset has too-small sequences! either filter out short sequences or reduce n-gram size."
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False

class UniformFrequencyHybridEncoder(AutoCompleteEncoder):
    def __init__(self, dataset, compression_rate=0.5, n_gram=2, removal_probability=0.5):
        self.uniform_encoder = UniformEncoder(removal_probability=removal_probability)
        self.frequency_encoder = FrequencyEncoder(dataset, compression_rate=compression_rate, n_gram=n_gram)
    def name(self):
        return ('UniformFrequencyHybridEncoder({}-gram, target_size:{})'
                .format(self.frequency_encoder.n_gram, self.frequency_encoder.compression_rate))
    def encode(self, s):
        if random.uniform(0,1) > 0.5:
            return self.uniform_encoder.encode(s)
        else:
            return self.frequency_encoder.encode(s)
    def encode_batch(self, b):
        if random.uniform(0,1) > 0.5:
            return self.uniform_encoder.encode_batch(b)
        else:
            return self.frequency_encoder.encode_batch(b)
    def is_optimizeable(self): return False

class RulesBasedEncoder(AutoCompleteEncoder):
    'Rules-based encoder baseline for Python'

    def __init__(self, whitespace=False):
        self.whitespace = whitespace
        rules_to_add = {' ': ''} if whitespace else None

        # create a dict with some default rules that I think we should have
        # then more additional rules can be added with rules_to_add dict
        # structure of dict ["long sequence" : 'shortened version']
        rules = {
            # python rules
            'var': 'vr',
            'list': 'ls',
            'range': 'rng',
            'lambda': 'lb',
            'enumerate': 'enm',
            'def': 'df',
            'return': 'ret',
            'for': 'fr',
            'value': 'val',
            'sorted': 'std',
            'reversed': 'rev',
            'True': 'T',
            'False': 'F',
            'condition': 'cd',
            'class': 'cl',
            'string': 'str',
            'split': 'spl',
            'format': 'fmt',
            'except': 'expt',
            'assert': 'ast',
            'collection': 'col',
            'None': 'N',
            'tuple': 'tup',
            'sort': 'srt',
            'continue': 'cnt',
            'break': 'bk',
            'global': 'glb',
            'while': 'whl',
            'import': 'imp',
            'raise': 'rs',
            'yield': 'yld',
            'print': 'pt',
            'self': 'sf',
            'elif': 'elf',
            # Java rules
            'abstract': 'abs',
            'boolean': 'bool',
            'catch': 'cth',
            'default': 'df',
            'double': 'dbl',
            'else': 'el',
            'extends': 'etx',
            'final': 'fnl',
            'float': 'flt',
            'implements': 'impl',
            'instanceof': 'itof',
            'interface': 'itf',
            'long': 'lng',
            'package': 'pkg',
            'private': 'pvt',
            'protected': 'ptd',
            'public': 'pbc',
            'short': 'sht',
            'static': 'stc',
            'super': 'spr',
            'switch': 'swt',
            'this': 't',
            'throws': 'trws',
            'throw': 'trw',
            'void': 'vd',
            'true': 'T',
            'false': 'F',
            'null': 'n',
            'System': 'sys',
            'java': 'jv',
            'util': 'utl',
            'String': 'str',
            "Array": 'arr',
            'ArrayList': 'arls',
            'HashMap': 'hmap',
            'List': 'ls',
            # Haskell rules
            'case': 'cs',
            'data': 'dt',
            'family': 'fm',
            'instance': 'it',
            'deriving': 'der',
            'forall': 'fa',
            'foreign': 'fgn',
            'hiding': 'hd',
            'infix': 'ifx',
            'infixl': 'ifxl',
            'infixr': 'ifxr',
            'module': 'mdl',
            'newtype': 'nt',
            'proc': 'pc',
            'qualified': 'ql',
            'type': 't',
            'where': 'wh',


        }
        self.rules = {**rules, **
                      rules_to_add} if (rules_to_add is not None) else rules
        self.rules_itemized = list(self.rules.items())

    def name(self):
        return 'RulesBasedEncoderPython(whitespace={})'.format(self.whitespace)

    def encode(self, s):
        # loop through list of keys, if match replace key with value from rules
        sent = s
        for rule, key in self.rules_itemized:
            if sent.find(rule) >= 0:
                sent = sent.replace(rule, key)
        return sent

    def encode_batch(self, b):
        return [self.encode(s) for s in b]

    def is_optimizeable(self):
        return False
