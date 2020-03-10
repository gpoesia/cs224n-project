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
          alphabet,
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
    encoder_learning_rate = parameters.get('encoder_learning_rate') or learning_rate / 10
    lambda_learning_rate = parameters.get('lambda_learning_rate') or 1e-2
    batch_size = parameters.get('batch_size') or 32
    init_scale = parameters.get('init_scale') or 0.1
    initial_encoder_bias = parameters.get('initial_encoder_bias') or 3.0
    initial_lambda = parameters.get('initial_lambda') or 30
    timeout = parameters.get('timeout')
    epochs = None if timeout in parameters else (parameters.get('epochs') or 1)
    verbose = parameters.get('verbose') or False
    log_every = parameters.get('log_every') or 100
    save_model_every_epoch = parameters.get('save_model_every_epoch') or False
    epsilon = parameters.get('epsilon') or 0.6

    training_set = dataset
    # validation_set = dataset['dev']

    train_losses = []

    if encoder.is_optimizeable():
        train_reconstruction_losses = []
        train_fraction_kept = []

    lambda_ = torch.tensor(initial_lambda, dtype=torch.float,
                           requires_grad=True, device=device)

    all_parameters_iter = lambda: itertools.chain(decoder.parameters(), alphabet.parameters()
                if alphabet.is_optimizeable() else iter([]))
    

    log = print if verbose else lambda *args: None
    decoder.to(device)

    optimizer_dec = torch.optim.Adam(all_parameters_iter(), lr=learning_rate)

    for p in all_parameters_iter():
        p.data.uniform_(-init_scale, init_scale)

    if encoder.is_optimizeable():
        encoder.to(device)
        optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=encoder_learning_rate)

        for p in encoder.parameters():
            p.data.uniform_(-init_scale, init_scale)

        encoder.output_proj.bias.data.fill_(initial_encoder_bias)

    begin_time = time.time()
    examples_processed = 0
    total_examples = epochs * batch_size * math.ceil(len(training_set) / batch_size) if epochs else math.inf
    if save_model_every_epoch:
        intermediate_models = []
        print('Saving model after every epoch')

    if encoder.is_optimizeable():
        print('Initial lambda:', lambda_.item())
        encoder.epsilon = epsilon
    s = time.time()
    e = 0
    while time.time() - s < timeout if timeout else e < epochs:
        for i in range((len(training_set) + batch_size) // batch_size):
            if (timeout and time.time() - s < timeout) or timeout is None:
                batch = random.sample(training_set, batch_size)
                lengths = torch.tensor([len(i) for i in batch], device=device)
                optimizer_dec.zero_grad()

                if lambda_.grad is not None:
                    lambda_.grad *= 0

                # If training with a non-neural encoder, just compute the average
                # prediction loss and optimize the decoder.
                if not encoder.is_optimizeable():
                    encoded_batch = encoder.encode_batch(batch)
                    per_prediction_loss = decoder(compressed=encoded_batch, alphabet=alphabet, expected=batch) #in training time
                    loss = (per_prediction_loss.sum(dim=1)/(lengths+1)).mean()
                else:
                    optimizer_enc.zero_grad()

                    batch_size = len(batch)
                    C = alphabet.encode_batch(batch) #(B,L,D)
                    C_indices = alphabet.encode_batch_indices(batch)
                    num_src_tokens = C.shape[1]


                    # (B,L)
                    encoded_batch_probs = encoder(encoded_batch=C, alphabet=alphabet, encoded_batch_indices=C_indices)

                    encoded_batch = torch.bernoulli(encoded_batch_probs) #(B,L)
                    encoded_batch.masked_fill_(C_indices == alphabet.padding_token_index(), 0)
                    num_input_tokens = (encoded_batch_probs > 0).sum(dim=1)

                    encoded_batch_strings = [''.join([batch[i][j]
                                            for j in range(len(batch[i]))
                                            if encoded_batch[i][j+1]])
                                            for i in range(batch_size)]

                    per_prediction_loss = decoder(
                        compressed=encoded_batch_strings, alphabet=alphabet, expected=batch).sum(dim=1)  # B

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
                    enc_likelihood = (key_likelihood_per_sample / (num_input_tokens - 1)).mean()

                    train_fraction_kept.append(avg_kept.item())
                    train_reconstruction_losses.append(reconstruction_loss.item())

                loss.backward()
                # import pdb; pdb.set_trace()
                optimizer_dec.step()

                # Maximize the loss over lambda.
                if encoder.is_optimizeable():
                    with torch.no_grad():
                        lambda_ += lambda_learning_rate * lambda_.grad

                    optimizer_enc.step()

                train_losses.append(loss.item())

                examples_processed += len(batch)

                if (len(train_losses) - 1) % log_every == 0:
                    time_elapsed = time.time() - begin_time
                    throughput = examples_processed / time_elapsed
                    remaining_seconds = int((total_examples - examples_processed) / throughput)
                    log('Epoch {} iteration {}: loss = {:.3f}, {}tp = {:.2f} lines/s, ETA {:02}h{:02}m{:02}s'.format(e, i, train_losses[-1],
                        (''
                        if not encoder.is_optimizeable()
                        else 'lambda: {:.3f}, % kept: {:.3f}, rec_loss: {:.3f}, enc_ll: {:.2f}, '.format(
                            lambda_.item(), avg_kept.item(), reconstruction_loss.item(),
                            enc_likelihood.item()
                            )),
                        throughput,
                        remaining_seconds // (60*60),
                        remaining_seconds // 60 % 60,
                        remaining_seconds % 60,
                        ))
            else:
<<<<<<< HEAD
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
                avg_likelihood = key_likelihood_per_sample.mean()
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
                     else 'lambda: {:.3f}, % kept: {:.3f}, rec_loss: {:.3f}, avg likelihood: {:.3f}'.format(
                         lambda_.item(), avg_kept.item(),
                         reconstruction_loss.item(), avg_likelihood.item())),
                    throughput,
                    remaining_seconds // (60*60),
                    remaining_seconds // 60 % 60,
                    remaining_seconds % 60,
                    ))

=======
                break
        e +=1
>>>>>>> da1b9f310b618190b8d2b826497bf0650f6c2930
        if save_model_every_epoch:
            intermediate_models.append((
                    (encoder.state_dict() if encoder.is_optimizeable() else None),
                    decoder.state_dict()
                    ))

    if save_model_every_epoch:
        return train_losses, intermediate_models

    if encoder.is_optimizeable():
        print('Final lambda:', lambda_.item())
    d = {'losses': train_losses, 'decoder_state_dict': decoder.state_dict()}
    if encoder.is_optimizeable():
        d['encoder_state_dict'] = encoder.state_dict()
        d['rec_loss'] = train_reconstruction_losses
        d['frac_kept'] = train_fraction_kept
    if alphabet.is_optimizeable():
        d['alphabet_state_dict'] = alphabet.state_dict()
    return d
