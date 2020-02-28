# Seq2Seq decoder model

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoCompleteDecoderModel(nn.Module):
    def __init__(self, alphabet, hidden_size=100, max_test_length=200):
        super().__init__()

        self.alphabet = alphabet
        self.encoder_lstm = nn.LSTM(alphabet.size(), hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTMCell(alphabet.size(), hidden_size)
        self.decoder_proj = nn.Linear(hidden_size, alphabet.size())
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

        output, final_state = self.encoder_lstm(C)

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
        next_input = (E_emb[:, 0]
                      if is_training
                      else self.alphabet.get_start_token().repeat(B, 1))
        last_output = None
        all_finished = False

        while not all_finished:
            decoder_state = self.decoder_lstm(next_input, decoder_state)
            last_output = self.decoder_proj(decoder_state[0])

            if is_training:
                predictions.append(last_output)
                next_input = E_emb[:, i + 1]
            else:
                # At test time, set next input to last predicted character
                # (greedy decoding).
                predictions = last_output.argmax(dim=1)
                finished[predictions == self.alphabet.end_token_index()] = 1

                for idx in (finished == 0).nonzero():
                    decoded_strings[idx].append(
                            self.alphabet.index_to_char(predictions[idx]))

                next_input = self.alphabet.encode_tensor_indices(predictions)

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
