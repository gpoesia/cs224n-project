# Train an end-to-end model

import torch
import random
import itertools
import time

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

    for e in range(epochs):
        for i in range((len(training_set) + batch_size) // batch_size):
            batch = random.sample(training_set, batch_size)

            optimizer.zero_grad()
            encoded_batch = encoder.encode_batch(batch)
            loss = decoder(encoded_batch, batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            examples_processed += len(batch)

            if (len(train_losses) - 1) % log_every == 0:
                time_elapsed = time.time() - begin_time
                throughput = examples_processed / time_elapsed
                log('Epoch {} iteration {}: loss = {:.3f}, tp = {:.2f} lines/s'.format(e, i, train_losses[-1], throughput))

    return train_losses
