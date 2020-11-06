import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noise import noisy

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)

def log_prob(z, mu, logvar):
    var = torch.exp(logvar)
    logp = - (z-mu)**2 / (2*var) - torch.log(2*np.pi*var) / 2
    return logp.sum(dim=1)

def loss_kl(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)


class F_VAE(nn.Module):
    # Overall architecture is built based on https://github.com/shentianxiao/text-autoencoders
    def __init__(self, vocab, embedding_dim, hidden_dim, layers_cnt, latent_dim, dropout,
                 learning_rate=0.0005, initrange=0.1):
        super().__init__()
        self.vocab = vocab
        self.embedding_size = embedding_dim
        self.embed = nn.Embedding(vocab.size, self.embedding_size)
        self.proj = nn.Linear(hidden_dim, vocab.size)

        self.embed.weight.data.uniform_(-initrange, initrange)
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim,
                               num_layers=layers_cnt,
                               dropout=dropout if layers_cnt > 1 else 0,
                               bidirectional=True)
        self.decoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim,
                               num_layers=layers_cnt,
                               dropout=dropout if layers_cnt > 1 else 0)
        self.h2mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.h2logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.z2emb = nn.Linear(latent_dim, embedding_dim)
        self.opt = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def flatten(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def encode(self, input):
        input = self.drop(self.embed(input))
        # |h_t| of shape (num_layers * num_directions, batch, hidden_size):
        #   tensor containing the hidden state h_t for t = seq_len.
        _, (h_t, _) = self.encoder(input)
        hidden_state = torch.cat([h_t[-2], h_t[-1]], 1)  # cause bidirectional
        return self.h2mu(hidden_state), self.h2logvar(hidden_state)

    def decode(self, z, input, hidden=None):
        input = self.drop(self.embed(input)) + self.z2emb(z)
        output, hidden = self.decoder(input, hidden)
        output = self.drop(output)
        logits = self.proj(output.view(-1, output.size(-1)))
        return logits.view(output.size(0), output.size(1), -1), hidden

    def forward(self, input, is_train=False):
        _input = noisy(self.vocab, input, *self.args.noise) if is_train else input
        mu, logvar = self.encode(_input)
        z = reparameterize(mu, logvar)
        logits, _ = self.decode(z, input)
        return mu, logvar, z, logits

    def loss_rec(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
            ignore_index=self.vocab.pad, reduction='none').view(targets.size())
        return loss.sum(dim=0)

    def step(self, losses):
        self.opt.zero_grad()
        losses['loss'].backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #nn.utils.clip_grad_norm_(self.parameters(), clip)
        self.opt.step()

    def nll_is(self, inputs, targets, m):
        """compute negative log-likelihood by importance sampling:
           p(x;theta) = E_{q(z|x;phi)}[p(z)p(x|z;theta)/q(z|x;phi)]
        """
        mu, logvar = self.encode(inputs)
        tmp = []
        for _ in range(m):
            z = reparameterize(mu, logvar)
            logits, _ = self.decode(z, inputs)
            v = log_prob(z, torch.zeros_like(z), torch.zeros_like(z)) - \
                self.loss_rec(logits, targets) - log_prob(z, mu, logvar)
            tmp.append(v.unsqueeze(-1))
        ll_is = torch.logsumexp(torch.cat(tmp, 1), 1) - np.log(m)
        return -ll_is

    def loss(self, losses):
        return losses['rec'] + self.args.lambda_kl * losses['kl']

    def autoenc(self, inputs, targets, is_train=False):
        mu, logvar, _, logits = self(inputs, is_train)
        return {'rec': self.loss_rec(logits, targets).mean(),
                'kl': loss_kl(mu, logvar)}
