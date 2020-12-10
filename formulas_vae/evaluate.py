import torch
import numpy as np


def encode(batches, order, model):
    z = []
    for inputs, _ in batches:
        mu, logsigma = model.encode(inputs)
        z.append(model.sample_z(mu, logsigma).detach().cpu().numpy())
    z_shape = (len(z), -1, z[0].shape[1])
    z = np.concatenate(z, axis=0)
    _, z = zip(*sorted(zip(order, z), key=lambda t: t[0]))
    z = np.array(list(z))
    return z.reshape(z_shape)


def decode(vocab, z, model, device, max_len):
    formulas = []
    for i in range(len(z)):
        outputs = generate(vocab, model, torch.tensor(z[i], device=device), max_len).t()
        for s in outputs:
            formulas.append([vocab.index_to_token[id] for id in s[1:]])
    formulas = [f[:f.index(vocab.eos_token)] if vocab.eos_token in f else f for f in formulas]
    return formulas


def generate(vocab, model, z, max_len):
    formulas = []
    x = torch.zeros(1, len(z), dtype=torch.long, device=z.device).fill_(vocab.sos_token_index)
    hidden = None
    for i in range(max_len):
        formulas.append(x)
        logits, hidden = model.decode(x, z, hidden)
        x = logits.argmax(dim=-1)
    return torch.cat(formulas)


def reconstruct(vocab, device, model, batches, order, max_len, out_file):
    z = encode(batches, order, model)
    reconstructed_formulas = decode(vocab, z, model, device, max_len)
    with open(out_file, 'w') as f:
        for formula in reconstructed_formulas:
            f.write(' '.join(formula) + '\n')
    with open(out_file + 'z', 'w') as f:
        for batch_z in z:
            for zi in batch_z:
                for zi_k in zi:
                    f.write('%f ' % zi_k)
                f.write('\n')

