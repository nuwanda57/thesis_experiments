import torch.nn.functional as F
import torch

import random
from copy import deepcopy


# Reconstruction error + KL divergence
def loss_function(logits, x, mu, logsigma, lambda_kl=1):
    reconstruction_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), reduction='none').view(x.size())
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp())
    return reconstruction_loss.sum(dim=0) + lambda_kl * KLD


def train(model, device, batches, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        indices = list(range(len(batches)))
        random.shuffle(indices)
        for idx in indices:
            optimizer.zero_grad()

            inputs = batches[idx]
            logits, mu, logsigma = model(inputs)

            loss = loss_function(logits, inputs, mu, logsigma)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print('loss', total_loss / len(indices))
