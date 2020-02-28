# Seq2Seq decoder model

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoCompleteDecoderModel(nn.Module):
    def __init__(self, alphabet, hidden_size=100, max_test_length=200, dropout_rate=0.2):
        super().__init__()

        self.alphabet = alphabet
        self.hidden_size = hidden_size
        self.encoder_lstm = nn.LSTM(alphabet.size(), hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTMCell(hidden_size + alphabet.size(), hidden_size)
        self.output_proj = nn.Linear(2*hidden_size, hidden_size, alphabet.size())
        self.vocab_proj = nn.Linear(hidden_size, alphabet.size())
        self.attention_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_test_length = max_test_length

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
        C_padding_tokens = torch.zeros((C.shape[0], C.shape[1]),
                                        dtype=torch.long,
                                        device=C.device)

        for i in range(B):
            # Everything after string (+ 2 tokens for begin and end) gets set to 1.
            # Used to make attention ignore padding tokens.
            C_padding_tokens[i][len(compressed[i]) + 2:] = 1

        encoder_hidden_states, final_state = self.encoder_lstm(C)

        if is_training:
            E = self.alphabet.encode_batch_indices(expected)
            E_emb = self.alphabet.encode_batch(expected)
            predictions = []

        finished = torch.zeros(B)
        decoded_strings = [[] for _ in range(B)]

        # The encoder LSTM state's first dimension is the layer index.
        # Since LSTMCell is single-layer, we need to get only the state of the
        # top-most layer (-1).
        decoder_state = (final_state[0][-1], final_state[1][-1])
        i = 0
        next_input = torch.cat([
            (E_emb[:, 0]
             if is_training
             else self.alphabet.get_start_token().repeat(B, 1)),
            torch.zeros((B, self.hidden_size),
                        dtype=torch.float,
                        device=self.alphabet.device)
        ], dim=1)

        last_output = None
        all_finished = False

        while not all_finished:
            (decoder_hidden, decoder_cell) = self.decoder_lstm(next_input, decoder_state)

            # decoder_hidden: (B, H)
            # encoder_hidden_states: (B, L, H)
            attention_queries = self.attention_proj(decoder_hidden) # (B, H)
            attention_scores = torch.squeeze(torch.bmm(encoder_hidden_states, # (B, L, H)
                                                       torch.unsqueeze(attention_queries, -1) # (B, H, 1)
                                                       ), 2) # -> (B, L)

            # Set attention scores to -infinity at padding tokens.
            attention_scores.data.masked_fill_(C_padding_tokens.bool(), -float('inf'))

            attention_d = F.softmax(attention_scores, dim=1)
            attention_result = torch.squeeze(torch.bmm(torch.unsqueeze(attention_d, 1),
                                             encoder_hidden_states), dim=1)
            U = torch.cat([decoder_hidden, attention_result], dim=1)
            V = self.output_proj(U)
            last_output = self.vocab_proj(self.dropout(torch.tanh(V)))

            if is_training:
                predictions.append(last_output)
                next_input = torch.cat([E_emb[:, i + 1], V], dim=1)
            else:
                # At test time, set next input to last predicted character
                # (greedy decoding).
                predictions = last_output.argmax(dim=1)
                finished[predictions == self.alphabet.end_token_index()] = 1

                for idx in (finished == 0).nonzero():
                    decoded_strings[idx].append(
                            self.alphabet.index_to_char(predictions[idx]))

                next_input = torch.cat([
                    self.alphabet.encode_tensor_indices(predictions),
                    attention_result
                ], dim=1)

            i += 1

            if is_training:
                all_finished = (i + 1 == E.shape[1])
            else:
                all_finished = i == self.max_test_length or finished.sum() == B

        if is_training:
            return (
                    nn.CrossEntropyLoss(ignore_index=self.alphabet.padding_token_index())
                    (torch.stack(predictions, dim=1).view((-1, self.alphabet.size())),
                     E[:, 1:].reshape((-1,)))
            )
        else:
            return [''.join(s) for s in decoded_strings]
