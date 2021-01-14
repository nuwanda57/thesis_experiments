import torch
import torch.nn as nn

import numpy as np

from collections import namedtuple

import formulas_vae_v2.formula_utils as my_formula_utils


ModelParams = namedtuple('ModelParams', [
    'vocab_size', 'token_embedding_dim', 'hidden_dim', 'encoder_layers_cnt', 'decoder_layers_cnt',
    'latent_dim', 'vocab', 'device'])
ModelParams.__new__.__defaults__ = (None,) * len(ModelParams._fields)


class FormulaVARE(nn.Module):
    # Overall model architecture is based on but not identical to https://github.com/shentianxiao/text-autoencoders.

    def __init__(self, model_params):
        super().__init__()

        self.encoder = nn.LSTM(model_params.token_embedding_dim, model_params.hidden_dim,
                               model_params.encoder_layers_cnt, dropout=0, bidirectional=True)
        self.decoder = nn.LSTM(model_params.token_embedding_dim, model_params.hidden_dim,
                               model_params.decoder_layers_cnt, dropout=0)

        self.embedding = nn.Embedding(model_params.vocab_size, model_params.token_embedding_dim)
        self.linear = nn.Linear(model_params.hidden_dim, model_params.vocab_size)
        self.drop = nn.Dropout(0.1)

        self.hidden_to_mu = nn.Linear(model_params.hidden_dim * 2, model_params.latent_dim)
        self.hidden_to_logsigma = nn.Linear(model_params.hidden_dim * 2, model_params.latent_dim)
        self.z_to_embedding = nn.Linear(model_params.latent_dim, model_params.token_embedding_dim)

        self.latent_dim = model_params.latent_dim
        self.hidden_dim = model_params.hidden_dim

        self.vocab = model_params.vocab
        self.device = model_params.device

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

    def build_ordered_latents(self, batches, order, strategy):
        assert strategy in ['mu', 'sample'], 'wrong strategy'
        z = []
        for inputs, _ in batches:
            mu, logsigma = self.encode(inputs)
            if strategy == 'sample':
                zi = self.sample_z(mu, logsigma).detach().cpu().numpy()
            elif strategy == 'mu':
                zi = mu.detach().cpu().numpy()
            else:
                raise 42
            z.append(zi)
        # z_shape = (len(z), -1, z[0].shape[1])
        batch_size = z[0].shape[1]
        z = np.concatenate(z, axis=0)
        _, z = zip(*sorted(zip(order, z), key=lambda t: t[0]))
        z = np.array(list(z))
        i = 0
        new_z = []
        while i < len(z):
            new_z.append(torch.tensor(z[i: i + batch_size]))
            i += batch_size
        return new_z
        # z: (batches, z_in_batch, latent_dim)
        # return z.reshape(z_shape)

    def reconstruct(self, batches, order, max_len, out_file=None, strategy='sample'):
        z = self.build_ordered_latents(batches, order, strategy=strategy)
        # z: (batches, z_in_batch, latent_dim)
        encoded_formulas = self.reconstruct_encoded_formulas_from_latent(z, max_len)
        # encoded_formulas: (total_formula_count, max_len)
        reconstructed_formulas = []
        for e_formula in encoded_formulas:
            reconstructed_formulas.append([self.vocab.index_to_token[id] for id in e_formula[1:]])
        reconstructed_formulas = [
            f[:f.index(self.vocab.eos_token)] if self.vocab.eos_token in f else f for f in reconstructed_formulas]
        zs = [zi for batch_z in z for zi in batch_z]

        for i in range(len(reconstructed_formulas)):
            reconstructed_formulas[i] = my_formula_utils.unify_tokens_into_numbers(reconstructed_formulas[i])

        if out_file is not None:
            with open(out_file, 'w') as f:
                for formula in reconstructed_formulas:
                    f.write(' '.join(formula) + '\n')
            with open(out_file + 'z', 'w') as f:
                for zi in zs:
                    for zi_k in zi:
                        f.write('%f ' % zi_k)
                    f.write('\n')
        return reconstructed_formulas, zs

    def reconstruct_encoded_formulas_from_latent(self, z_batched, max_len):
        # z_batched: (batches, z_in_batch, latent_dim)
        formulas = []
        for z in z_batched:
            formulas_in_batch = []
            # z: (z_in_batch, latent_dim)
            x = torch.zeros(1, len(z), dtype=torch.long, device=self.device).fill_(self.vocab.sos_token_index)
            # x: (1, z_in_batch)
            hidden = None
            for i in range(max_len):
                formulas_in_batch.append(x)
                logits, hidden = self.decode(x, torch.tensor(z, device=self.device), hidden)
                x = logits.argmax(dim=-1)
            # formulas_in_batch [[[f1_0, f2_0, ..]], [[f1_1, f2_1, ..]], ..] -> [[f1_0, f1_1, ..], [f2_0, f2_1, ..], ..]
            formulas_in_batch = torch.cat(formulas_in_batch, 0).T
            for f in formulas_in_batch:
                formulas.append(f)
        return formulas

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
