import torch.nn as nn
import torch


# https://arxiv.org/pdf/1412.6581.pdf
class FormulasVARE(nn.Module):
    def __init__(self, vocab_size, token_embedding_dim, hidden_dim, encoder_layers_cnt, dropout, bidirectional_encoder,
                 decoder_layers_cnt, latent_dim):
        super().__init__()
        self.token_embedding_dim = token_embedding_dim
        self.embedding = nn.Embedding(vocab_size, token_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional_encoder = bidirectional_encoder
        self.encoder = nn.LSTM(token_embedding_dim, hidden_dim, encoder_layers_cnt,
            dropout=dropout, bidirectional=bidirectional_encoder)
        self.decoder = nn.LSTM(token_embedding_dim, hidden_dim, decoder_layers_cnt, dropout=dropout)
        if self.bidirectional_encoder:
            self.hidden_to_mu = nn.Linear(hidden_dim * 2, latent_dim)
            self.hidden_to_logsigma = nn.Linear(hidden_dim * 2, latent_dim)
        else:
            self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
            self.hidden_to_logsigma = nn.Linear(hidden_dim * 2, latent_dim)
        self.latent_to_embedding = nn.Linear(latent_dim, token_embedding_dim)

        self.activation1 = nn.Tanh()

        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.activation2 = nn.Softmax()

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
        x = self.dropout(x)
        # hidden_state: (formula_len, batch_size, hidden_dim)
        _, (hidden_state, _) = self.encoder(x)
        if self.bidirectional_encoder:
            hidden_state = torch.cat([hidden_state[-2], hidden_state[-1]], 1)
            # hidden_state: (batch_size, hidden_dim * 2)
        else:
            hidden_state = hidden_state[-1]
            # hidden_state: (batch_size, hidden_dim)
        mu = self.hidden_to_mu(hidden_state)
        # mu: (batch_size, latent_dim)
        logsigma = self.hidden_to_logsigma(hidden_state)
        # logsigma: (batch_size, latent_dim)
        return mu, logsigma

    # z - latent representation
    def decode(self, x, z):
        z_emb = self.activation1(self.latent_to_embedding(z))
        x = self.embedding(x)
        x = self.dropout(x)
        x = x + z_emb
        x, hidden = self.decoder(x, None)
        x = self.dropout(x)
        x = self.linear(x)
        logits = self.activation2(x)
        # logits: (formula_len, batch_size, vocab_size)
        return logits

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.sample_z(mu, logsigma)
        # z: (batch_size, latent_dim)
        logits = self.decode(x, z)
        # logits: (formula_len, batch_size, vocab_size)
        return logits, mu, logsigma, z

    def generate_greedy(self, z, max_len):
        tokens = []
        # len(z = batch size)
        x = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(0)
        for i in range(max_len):
            logits = self.decode(x, z)
            x = logits.argmax(dim=-1)
            tokens.append(x)
        return torch.cat(tokens)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
