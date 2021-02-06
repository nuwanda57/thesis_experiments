import formulas_vae_v4.formula_config as my_formula_config

import torch
import random
import torch.nn.functional as F
import numpy as np


# Reconstruction error + KL divergence
def loss_function(logits, targets, mu, logsigma):
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=my_formula_config.TOKEN_TO_INDEX[my_formula_config.PADDING], reduction='none').view(targets.size())
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) / len(mu)
    # reconstruction_loss: (formula_dim, batch_size), so we take sum over all torens and mean over formulas in batch
    return reconstruction_loss.sum(dim=0).mean(), KLD


def evaluate(model, batches):
    model.eval()
    kl_losses, rec_losses = [], []
    with torch.no_grad():
        for inputs, targets in batches:
            logits, mu, logsigma, z = model(inputs)
            rec, kl = loss_function(logits, targets, mu, logsigma)
            kl_losses.append(kl.item())
            rec_losses.append(rec.item())
    loss = np.mean(rec_losses)
    return loss, np.mean(rec_losses), np.mean(kl_losses)


def run_epoch(model, optimizer, train_batches, valid_batches, epoch):
    print('Epoch %d' % epoch)
    kl_losses, rec_losses, losses = [], [], []
    model.train()
    indices = list(range(len(train_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        optimizer.zero_grad()
        inputs, targets = train_batches[idx]
        logits, mu, logsigma, z = model(inputs)
        rec, kl = loss_function(logits, targets, mu, logsigma)
        loss = rec
        loss.backward()
        optimizer.step()
        rec_losses.append(rec.item())
        losses.append(loss.item())
        kl_losses.append(kl.item())

    print('\t[training] batches count: %d' % len(indices))
    print('\t[training] loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % (
        np.mean(losses), np.mean(rec_losses), np.mean(kl_losses)))

    valid_losses = evaluate(model, valid_batches)
    print('\t[validation] loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % valid_losses)
