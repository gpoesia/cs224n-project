# Train an end-to-end model

import torch
import random
import itertools
import time
import math

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

    train_losses = []
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

    for e in range(epochs):
        for i in range((len(training_set) + batch_size) // batch_size):
            batch = random.sample(training_set, batch_size)

            optimizer.zero_grad()
            encoded_batch = encoder.encode_batch(batch)
            per_prediction_loss = decoder(encoded_batch, batch)

            # If training with a non-neural encoder, just compute the average
            # prediction loss and optimize the decoder.
            if not encoder.is_optimizeable():
                loss = per_prediction_loss.mean()
            else:
                raise NotImplemented("End-to-end loss not implemented yet")

            loss.backward()

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
