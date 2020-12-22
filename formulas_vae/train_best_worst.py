import torch
import random
import torch.nn.functional as F
import numpy as np

import os

import formulas_vae.utils as my_utils


# Reconstruction error + KL divergence
def loss_function(logits, targets, mu, logsigma, vocab):
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=vocab.pad_token_index, reduction='none').view(targets.size())
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) / len(mu)
    # reconstruction_loss: (formula_dim, batch_size), so we take sum over all torens and mean over formulas in batch
    return reconstruction_loss.sum(dim=0).mean(), KLD


def evaluate(model, batches, vocab):
    model.eval()
    kl_losses, rec_losses = [], []
    with torch.no_grad():
        for inputs, targets in batches:
            logits, mu, logsigma, z = model(inputs)
            rec, kl = loss_function(logits, targets, mu, logsigma, vocab)
            kl_losses.append(kl.item())
            rec_losses.append(rec.item())
    loss = np.mean(rec_losses)
    return loss, np.mean(rec_losses), np.mean(kl_losses)


def update_train_dataset(old_train_path, new_train_path, vocab, model, batch_size, device, max_len,
                         xs, training_dir, choose_worst, fraction):
    train_batches, train_order = my_utils.build_ordered_batches(old_train_path, vocab, batch_size, device)
    rec_path = os.path.join(training_dir, 'rec')
    model.reconstruct(train_batches, train_order, max_len, rec_path, strategy='mu')
    mses = my_utils.reconstruction_mses(rec_path, old_train_path, xs)

    mse_with_inds = list(enumerate(mses))
    mse_with_inds.sort(key=lambda k: k[1], reverse=choose_worst)

    new_formula_indices = set([m[0] for m in mse_with_inds[:(int(len(mse_with_inds) * fraction))]])
    written = 0
    with open(old_train_path) as old_file, open(new_train_path, 'w') as new_file:
        for i, line in enumerate(old_file):
            if i in new_formula_indices:
                new_file.write(line)
                written += 1
                if written != len(new_formula_indices):
                    new_file.write('\n')


def train(vocab, model, optimizer, train_file_path, valid_batches, epochs, batch_size, max_len, device,
          log_interval=20, update_train_epochs=50, training_log_dir='training/', choose_worst=True):

    xs = np.linspace(0.0, 1.0, num=100)
    if not os.path.exists(training_log_dir):
        os.mkdir(training_log_dir)

    cur_train_path = train_file_path
    train_batches, _ = my_utils.build_ordered_batches(cur_train_path, vocab, batch_size, device)
    for epoch in range(epochs):
        print('Epoch %d' % (epoch + 1))
        kl_losses, rec_losses, losses = [], [], []
        model.train()
        indices = list(range(len(train_batches)))
        random.shuffle(indices)
        for i, idx in enumerate(indices):
            optimizer.zero_grad()
            inputs, targets = train_batches[idx]
            logits, mu, logsigma, z = model(inputs)
            rec, kl = loss_function(logits, targets, mu, logsigma, vocab)
            loss = rec
            loss.backward()
            optimizer.step()
            rec_losses.append(rec.item())
            losses.append(loss.item())
            kl_losses.append(kl.item())

            if (i + 1) % log_interval == 0:
                print('\t[training] %d/%d' % (i + 1, len(indices)), end=' ')
                print('loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % (
                    np.mean(losses), np.mean(rec_losses), np.mean(kl_losses)))

        if (epoch + 1) % update_train_epochs == 0:
            print('updating train dataset')
            new_train_path = os.path.join(training_log_dir, 'train_%d.txt' % (i + 1))
            update_train_dataset(cur_train_path, new_train_path, vocab, model, batch_size, device, max_len,
                                 xs, training_log_dir, choose_worst, 0.9)
            cur_train_path = new_train_path
            train_batches, _ = my_utils.build_ordered_batches(cur_train_path, vocab, batch_size, device)

        valid_losses = evaluate(model, valid_batches, vocab)
        print('\t[validation] loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % valid_losses)
