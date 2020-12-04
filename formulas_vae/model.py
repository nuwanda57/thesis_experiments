import torch
import torch.nn as nn


class FormulaVARE(nn.Module):
    # Overall model architecture is based on but not identical to https://github.com/shentianxiao/text-autoencoders.

    def __init__(self, vocab_size, token_embedding_dim, hidden_dim, encoder_layers_cnt, decoder_layers_cnt, latent_dim):
        super().__init__()

        self.encoder = nn.LSTM(token_embedding_dim, hidden_dim, encoder_layers_cnt, dropout=0, bidirectional=True)
        self.decoder = nn.LSTM(token_embedding_dim, hidden_dim, decoder_layers_cnt, dropout=0)

        self.embedding = nn.Embedding(vocab_size, token_embedding_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.drop = nn.Dropout(0.1)

        self.hidden_to_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.hidden_to_logsigma = nn.Linear(hidden_dim * 2, latent_dim)
        self.z_to_embedding = nn.Linear(latent_dim, token_embedding_dim)

        self._reset_parameters()

    @staticmethod
    def sample_z(mu, logsigma):
        # https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73:
        #  if z is a random variable following a Gaussian distribution with
        #  mean g(x) and with covariance H(x)=h(x).h^t(x) then it can be expressed as
        #  z = h(x)*a + g(x) where a ~ N(0, I).
        # https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def encode(self, x):
        # x: (formula_len, batch_size)
        x = self.embedding(x)
        # x: (formula_len, batch_size, embedding_dim)
        x = self.drop(x)
        # hidden_state: (formula_len, batch_size, hidden_dim)
        _, (hidden_state, _) = self.encoder(x)
        hidden_state = torch.cat([hidden_state[-2], hidden_state[-1]], 1)
        mu = self.hidden_to_mu(hidden_state)
        # mu: (batch_size, latent_dim)
        logsigma = self.hidden_to_logsigma(hidden_state)
        # logsigma: (batch_size, latent_dim)
        return mu, logsigma

    # z - latent representation
    def decode(self, x, z, hidden=None):
        # x: (formula_len, batch_size)
        z_emb = self.z_to_embedding(z)
        x = self.embedding(x)
        x = self.drop(x)
        # x: (formula_len, batch_size, embedding_dim)
        x = x + z_emb
        # x: (formula_len, batch_size, embedding_dim)
        x, hidden = self.decoder(x, hidden)
        # x: (formula_len, batch_size, embedding_dim)
        x = self.drop(x)
        logits = self.linear(x)
        # logits = self.activation2(x)
        # logits: (formula_len, batch_size, vocab_size)
        return logits, hidden

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.sample_z(mu, logsigma)
        # z: (batch_size, latent_dim)
        logits, _ = self.decode(x, z)
        # logits: (formula_len, batch_size, vocab_size)
        return logits, mu, logsigma, z

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
