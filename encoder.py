from base import AutoCompleteEncoder
import random
import torch
import torch.nn as nn

class NeuralEncoder(nn.Module):
    'Encodes a string by removing characters.'
    def __init__(self, alphabet, epsilon, hidden_size=100):
        super().__init__()

        self.alphabet = alphabet
        self.epsilon = epsilon
        self.hidden_size = hidden_size
        # what are input dimensions
        self.encoder_lstm = nn.LSTM(alphabet.size(), hidden_size, batch_first=True)
        self.output_proj = nn.Linear(hidden_size, 1)
        print("hello darkness")

    def name(self):
        return 'Encoder({:.2f})'.format(self.epsilon)

    def encode(self, s):
        pass

    def forward(self, b):
        """ encode is forward """
        batch_size = len(b)
        C = self.alphabet.encode_batch(b) #(B,L,D)
        encoder_hidden_states, final_state = self.encoder_lstm(C) #(B,L,H), H
        prob_of_drop = torch.sigmoid(self.output_proj(encoder_hidden_states)).squeeze(2) # (B,L,1)
        return prob_of_drop

    def is_optimizeable(self):
        return True
