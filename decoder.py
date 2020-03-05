# Seq2Seq decoder model

import torch
import torch.nn as nn
import torch.nn.functional as F

# Types of copy mechanism
COPY_CLASSIC = 'classic'
COPY_SUBSEQ = 'subseq'

class AutoCompleteDecoderModel(nn.Module):
    def __init__(self, alphabet, hidden_size=100, max_test_length=200, dropout_rate=0.2,
                 copy=None):
        super().__init__()

        self.alphabet = alphabet
        self.hidden_size = hidden_size
        self.encoder_lstm = nn.LSTM(alphabet.size(), hidden_size, batch_first=True, bidirectional=True)
        self.decoder_lstm = nn.LSTMCell(hidden_size + alphabet.size(), hidden_size)
        self.h_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.c_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.attention_proj = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.output_proj = nn.Linear(3*hidden_size, hidden_size, bias=False)
        self.vocab_proj = nn.Linear(hidden_size, alphabet.size(), bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_test_length = max_test_length
        self.copy = copy

        if copy == COPY_CLASSIC:
            self.p_gen_context_weight = nn.Parameter(torch.zeros(2*hidden_size))
            self.p_gen_input_weight = nn.Parameter(
                    torch.zeros(alphabet.size() + hidden_size))
            self.p_gen_state_h_weight = nn.Parameter(torch.zeros(hidden_size))
            self.p_gen_state_c_weight = nn.Parameter(torch.zeros(hidden_size))
            self.p_gen_bias = nn.Parameter(torch.zeros(1))

    def forward(self, compressed, expected=None):
        '''Forward pass, for test time if expected is None, otherwise for training.

        - @param compressed: list of compressed strings.
        - @param expected: list of expected (expanded) output string.
        - @returns the loss term, if expected is passed. Otherwise, returns
                   a list with one string with the predictions for each input
                   compressed string.

        Encodes the compressed input sentences in C and decodes them using
        the sentences in T as the target.'''

        B = len(compressed)
        is_training = expected is not None

        C = self.alphabet.encode_batch(compressed)
        C_indices = self.alphabet.encode_batch_indices(compressed)
        C_padding_tokens = torch.zeros((C.shape[0], C.shape[1]),
                                        dtype=torch.long,
                                        device=C.device)

        for i in range(B):
            # Everything after string (+ 2 tokens for begin and end) gets set to 1.
            # Used to make attention ignore padding tokens.
            C_padding_tokens[i][len(compressed[i]) + 2:] = 1

        encoder_hidden_states, (enc_hn, enc_cn) = self.encoder_lstm(C)
        decoder_state = (self.h_proj(enc_hn.transpose(0, 1).reshape(B, -1)),
                         self.c_proj(enc_cn.transpose(0, 1).reshape(B, -1)))

        if is_training:
            E = self.alphabet.encode_batch_indices(expected)
            E_emb = self.alphabet.encode_batch(expected)
            predictions = []

            if self.copy == COPY_SUBSEQ:
                copy_positions = torch.stack([
                    get_copy_positions(full, c, E.shape[1] - 1)
                    for full, c in zip(expected, compressed)
                ], dim=0)

                E[copy_positions] = self.alphabet.copy_token_index()

        if not is_training and self.copy == COPY_SUBSEQ:
            copy_counters = [0 for _ in range(B)]

        finished = torch.zeros(B)
        decoded_strings = [[] for _ in range(B)]

        i = 0
        next_input = torch.cat([
            (self.alphabet.get_start_token().repeat(B, 1)),
            torch.zeros((B, self.hidden_size),
                        dtype=torch.float,
                        device=self.alphabet.device)
        ], dim=1)

        last_output = None
        all_finished = False

        encoder_hidden_states_proj = self.attention_proj(encoder_hidden_states)

        copy_classic = self.copy is COPY_CLASSIC

        while not all_finished:
            decoder_state = (decoder_hidden, decoder_cell) = self.decoder_lstm(next_input, decoder_state)

            # decoder_hidden: (B, H)
            # encoder_hidden_states: (B, L, H)
            attention_scores = torch.squeeze(torch.bmm(encoder_hidden_states_proj, # (B, L, H)
                                                       torch.unsqueeze(decoder_hidden, -1) # (B, H, 1)
                                                       ), 2) # -> (B, L)

            # Set attention scores to -infinity at padding tokens.
            attention_scores.data.masked_fill_(C_padding_tokens.bool(), -float('inf'))

            attention_d = F.softmax(attention_scores, dim=1)
            attention_result = torch.squeeze(torch.bmm(torch.unsqueeze(attention_d, 1),
                                             encoder_hidden_states), dim=1)
            U = torch.cat([decoder_hidden, attention_result], dim=1)
            V = self.output_proj(U)
            timestep_out = self.dropout(torch.tanh(V))
            last_output = F.softmax(self.vocab_proj(timestep_out), dim=1)

            if copy_classic:
                p_gen = torch.sigmoid(
                            broadcast_dot(attention_result, self.p_gen_context_weight) +
                            broadcast_dot(next_input, self.p_gen_input_weight) +
                            broadcast_dot(decoder_hidden, self.p_gen_state_h_weight) +
                            broadcast_dot(decoder_cell, self.p_gen_state_c_weight) +
                            self.p_gen_bias
                        ).view((-1, 1))

                last_output = (last_output * p_gen).scatter_add(1, C_indices, attention_d * (1 - p_gen))

            if is_training:
                predictions.append(last_output)
                next_input = torch.cat([E_emb[:, i + 1], timestep_out], dim=1)
            else:
                # At test time, set next input to last predicted character
                # (greedy decoding).
                predictions = last_output.argmax(dim=1)
                finished[predictions == self.alphabet.end_token_index()] = 1

                for idx in (finished == 0).nonzero():
                    copy = (self.copy is COPY_SUBSEQ and
                               predictions[idx] == self.alphabet.copy_token_index())

                    decoded_strings[idx].append(
                            self.alphabet.index_to_char(predictions[idx])
                            if not copy
                            else (compressed[idx][copy_counters[idx]]
                                  if copy_counters[idx] < len(compressed[idx])
                                  else '')
                            )

                    if copy:
                        copy_counters[idx] += 1

                next_input = torch.cat([
                    self.alphabet.encode_tensor_indices(predictions),
                    timestep_out,
                ], dim=1)

            i += 1

            if is_training:
                all_finished = (i + 1 == E.shape[1])
            else:
                all_finished = i == self.max_test_length or finished.sum() == B

        if is_training:
            predictions = torch.stack(predictions, dim=1)
            return (
                    F.nll_loss(
                        predictions.transpose(1, 2).log(),
                        E[:, 1:],
                        ignore_index=self.alphabet.padding_token_index(),
                        reduction='none')
            )
        else:
            return [''.join(s) for s in decoded_strings]

def get_copy_positions(full_string, subseq, pad_to_length=0):
    '''Computes the sequence of copies and insertions needed to get to `full_string` from `subseq`.

    Example: to get from acf to abcdef, we need to copy a, insert b, copy c, insert d, insert e, copy f.

    @returns a boolean tensor of length max(len(full_string), pad_to_length),
    where each element is either True if the corresponding character will be copied
    from the input or False if it should be inserted. In case of multiple solutions,
    it always prefers to copy first.'''

    ans = torch.zeros(1 + max(len(full_string), pad_to_length), dtype=torch.bool)
    copied = 0

    for i, c in enumerate(full_string):
        if copied < len(subseq) and subseq[copied] == c:
            ans[i+1] = True
            copied += 1
        i += 1

    return ans

def broadcast_dot(m, v):
    'torch.dot() broadcasting version'
    return m.mm(v.view(-1, 1)).squeeze(1)
