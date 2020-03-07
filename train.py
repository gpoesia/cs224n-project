# Train an end-to-end model

import random
import itertools
import time
import math

import torch
import torch.nn.functional as F

def bce_loss_per_token(input, target, alphabet, c_indices):
    loss = F.binary_cross_entropy(input, target, reduction="none")
    loss.masked_fill_(target.ge(0).bitwise_not(), 0)  # ignore_index=-1
    indices = alphaencode_batch_indices
    loss.masked_fill_(c_indices == alphabet.PADDING_INDEX, 0)
    return loss

def train(encoder,
          decoder,
          dataset,
          parameters,
          device):
    '''
        Trains the end-to-end model using the specified parameters.

        Parameters:
        - batch_size,
        - learning_rate=1e-,
        - init_scale,
        - epochs
    '''

    learning_rate = parameters.get('learning_rate') or 1e-2
    batch_size = parameters.get('batch_size') or 32
    init_scale = parameters.get('init_scale') or 0.1
    epochs = parameters.get('epochs') or 1
    verbose = parameters.get('verbose') or False
    log_every = parameters.get('log_every') or 100
    save_model_every_epoch = parameters.get('save_model_every_epoch') or False

    training_set = dataset['train']
    validation_set = dataset['dev']
    alphabet = encoder.alphabet

    train_losses = []
    lam = torch.tensor(10, dtype=torch.float, requires_grad=True)
    all_parameters_iter = lambda: (
            itertools.chain(encoder.parameters(), decoder.parameters())
            if encoder.is_optimizeable()
            else decoder.parameters())

    log = print if verbose else lambda *args: None
    decoder.to(device)

    optimizer = torch.optim.Adam(all_parameters_iter(), lr=learning_rate)

    for p in all_parameters_iter():
        p.data.uniform_(-init_scale, init_scale)

    #print("lam: ", lam)

    begin_time = time.time()
    examples_processed = 0
    total_examples = epochs * batch_size * math.ceil(len(training_set) / batch_size)

    if save_model_every_epoch:
        intermediate_models = []
        print('Saving model after every epoch')

    for e in range(epochs):
        for i in range((len(training_set) + batch_size) // batch_size):
            batch = random.sample(training_set, batch_size)

            optimizer.zero_grad()

            print("batch: ", batch)
            # If training with a non-neural encoder, just compute the average
            # prediction loss and optimize the decoder.
            if not encoder.is_optimizeable():
                encoded_batch = encoder.encode_batch(batch)
                per_prediction_loss = decoder(encoded_batch, batch) #in training time
                loss = per_prediction_loss.mean()
            else:
                batch_size = len(batch)
                C = alphabet.encode_batch(batch) #(B,L,D)
                C_indices = alphabet.encode_batch_indices(batch)
                num_src_tokens = C.shape[1]
                encoded_batch_probs = encoder(C) #(B,L)
                print("encoded_batch_probs: ", encoded_batch_probs)
                encoded_batch = torch.bernoulli(encoded_batch_probs) #(B,L)
                encoded_batch_strings = [''.join([batch[i][j] for j in range(len(batch[i])) if encoded_batch[i][j+1]])
                                         for i in range(batch_size)]
                print("encoded_batch_strings:", encoded_batch_strings)
                per_prediction_loss = decoder(encoded_batch_strings, batch).sum(dim=1) #B
                print("per_prediction_loss: ", per_prediction_loss)
                reward_per_sample = -(lam * (per_prediction_loss - encoder.epsilon)
                                      + encoded_batch.sum(dim=1)).mean() #B

                key_likelihood_per_token = - bce_loss_per_token(
                    encoded_batch_probs.view(-1),
                    encoded_batch.float().view(-1), alphabet, C_indices)
                key_likelihood_per_sample = torch.sum(
                    key_likelihood_per_token.view(batch_size, num_src_tokens),
                    dim=1).div(num_src_tokens)

                encoder_loss = - (key_likelihood_per_sample *
                                reward_per_sample).sum().div(batch_size)



            loss.backward()
            print('grad: ', decoder.attention_proj.weight._grad)
            optimizer.step()
            train_losses.append(loss.item())

            examples_processed += len(batch)

            if (len(train_losses) - 1) % log_every == 0:
                time_elapsed = time.time() - begin_time
                throughput = examples_processed / time_elapsed
                remaining_seconds = int((total_examples - examples_processed) / throughput)
                log('Epoch {} iteration {}: loss = {:.3f}, tp = {:.2f} lines/s, ETA {:02}h{:02}m{:02}s'.format(e, i, train_losses[-1], throughput,
                    remaining_seconds // (60*60),
                    remaining_seconds // 60 % 60,
                    remaining_seconds % 60,
                    ))

        if save_model_every_epoch:
            intermediate_models.append((
                    (encoder.state_dict() if encoder.is_optimizeable() else None),
                    decoder.state_dict()
                    ))

    if save_model_every_epoch:
        return train_losses, intermediate_models

    return train_losses
