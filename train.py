# Train an end-to-end model

import random
import itertools
import time
import math

import torch
import torch.nn.functional as F

def bce_loss_per_token(input, target, alphabet, c_indices):
    loss = F.binary_cross_entropy(input, target, reduction="none")
    loss.masked_fill_(c_indices.view(-1) == alphabet.padding_token_index(), 0)
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
    lambda_learning_rate = parameters.get('lambda_learning_rate') or 1e-2
    batch_size = parameters.get('batch_size') or 32
    init_scale = parameters.get('init_scale') or 0.1
    initial_lambda = parameters.get('initial_lambda') or 30
    epochs = parameters.get('epochs') or 1
    verbose = parameters.get('verbose') or False
    log_every = parameters.get('log_every') or 100
    save_model_every_epoch = parameters.get('save_model_every_epoch') or False
    epsilon = parameters.get('epsilon') or 0.6

    training_set = dataset['train']
    validation_set = dataset['dev']
    alphabet = decoder.alphabet

    train_losses = []

    if encoder.is_optimizeable():
        train_reconstruction_losses = []
        train_fraction_kept = []

    lambda_ = torch.tensor(initial_lambda, dtype=torch.float, requires_grad=True)

    all_parameters_iter = lambda: (
            itertools.chain(encoder.parameters(), decoder.parameters())
            if encoder.is_optimizeable()
            else decoder.parameters())

    log = print if verbose else lambda *args: None
    decoder.to(device)

    optimizer = torch.optim.Adam(all_parameters_iter(), lr=learning_rate)

    for p in all_parameters_iter():
        p.data.uniform_(-init_scale, init_scale)

    begin_time = time.time()
    examples_processed = 0
    total_examples = epochs * batch_size * math.ceil(len(training_set) / batch_size)
    if save_model_every_epoch:
        intermediate_models = []
        print('Saving model after every epoch')

    if encoder.is_optimizeable():
        print('Initial lambda:', lambda_)
        encoder.epsilon = epsilon

    for e in range(epochs):
        for i in range((len(training_set) + batch_size) // batch_size):
            batch = random.sample(training_set, batch_size)

            optimizer.zero_grad()

            if lambda_.grad is not None:
                lambda_.grad *= 0

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

                encoded_batch_probs = encoder(C, C_indices) #(B,L)

                encoded_batch = torch.bernoulli(encoded_batch_probs) #(B,L)
                encoded_batch.masked_fill_(C_indices == alphabet.padding_token_index(), 0)
                num_input_tokens = (encoded_batch_probs > 0).sum(dim=1)

                encoded_batch_strings = [''.join([batch[i][j]
                                         for j in range(len(batch[i]))
                                         if encoded_batch[i][j+1]])
                                         for i in range(batch_size)]

                per_prediction_loss = decoder(encoded_batch_strings, batch).sum(dim=1) #B

                kept_tokens = encoded_batch.sum(dim=1)

                reward_per_sample = (lambda_ * (per_prediction_loss - epsilon)
                                      + kept_tokens) / num_input_tokens #B

                key_likelihood_per_token = - bce_loss_per_token(
                    encoded_batch_probs.view(-1),
                    encoded_batch.detach().float().view(-1), alphabet, C_indices)

                key_likelihood_per_sample = torch.sum(
                    key_likelihood_per_token.view(batch_size, num_src_tokens),
                    dim=1) # / kept_tokens

                # print('Batch:', batch)
                # print('encoded_batch:', encoded_batch)
                # print('encoded_batch_probs:', encoded_batch_probs)
                # print('encoded_batch_probs shape:', encoded_batch_probs.shape)
                # print('Per prediction loss:', per_prediction_loss)
                # print('Kept tokens:', kept_tokens)
                # print('Reward per sample:', reward_per_sample)
                # print('Key likelihood per token:', key_likelihood_per_token)
                # print('Key likelihood per sample:', key_likelihood_per_sample)

                loss = -1.0 * (key_likelihood_per_sample * reward_per_sample).mean()

                avg_kept = ((kept_tokens - 2) / (num_input_tokens - 2)).mean()
                reconstruction_loss = (per_prediction_loss / (num_input_tokens - 1)).mean()

                train_fraction_kept.append(avg_kept)
                train_reconstruction_losses.append(reconstruction_loss)

            loss.backward()

            optimizer.step()

            # Maximize the loss over lambda.
            if encoder.is_optimizeable():
                with torch.no_grad():
                    lambda_ += lambda_learning_rate * lambda_.grad

            train_losses.append(loss.item())

            examples_processed += len(batch)

            if (len(train_losses) - 1) % log_every == 0:
                time_elapsed = time.time() - begin_time
                throughput = examples_processed / time_elapsed
                remaining_seconds = int((total_examples - examples_processed) / throughput)
                log('Epoch {} iteration {}: loss = {:.3f}, {}tp = {:.2f} lines/s, ETA {:02}h{:02}m{:02}s'.format(e, i, train_losses[-1],
                    (''
                     if not encoder.is_optimizeable()
                     else 'lambda: {:.3f}, % kept: {:.3f}, rec_loss: {:.3f}, '.format(
                         lambda_.item(), avg_kept.item(), reconstruction_loss.item())),
                    throughput,
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

    if encoder.is_optimizeable():
        print('Final lambda:', lambda_)

    return (train_losses
            if not encoder.is_optimizeable()
            else (train_losses, train_reconstruction_losses, train_fraction_kept))
