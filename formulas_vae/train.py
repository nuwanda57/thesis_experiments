import torch.nn.functional as F
import torch

import random
from copy import deepcopy


# Reconstruction error + KL divergence
def loss_function(logits, x, mu, logsigma):
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), x.view(-1),
        reduction='none').view(x.size())
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    # reconstruction_loss: (formula_dim, batch_size), so we take sum over all torens and mean over formulas in batch
    return reconstruction_loss.mean(dim=0).mean(), KLD


def train(model, device, batches, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_rec_loss = 0
        total_kld = 0
        indices = list(range(len(batches)))
        random.shuffle(indices)
        steps_cnt = 0
        for idx in indices:
            optimizer.zero_grad()

            inputs = batches[idx]
            logits, mu, logsigma, _ = model(inputs)

            rec_loss, kld = loss_function(logits, inputs, mu, logsigma)
            loss = rec_loss + kld
            loss.backward()
            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kld += kld.item()

            optimizer.step()
            if steps_cnt % 100 == 0:
              print('step_loss: ', loss.item(), inputs.shape)
              print('rec_loss: ', rec_loss.item(), inputs.shape)
              print('kld_loss: ', kld.item(), inputs.shape)
            steps_cnt += 1
        print('loss', total_loss / len(indices))
        print('rec_loss', total_rec_loss / len(indices))
        print('kld_loss', total_kld / len(indices))
