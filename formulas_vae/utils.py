import torch


def build_single_batch_from_formulas_list(formulas_list, vocab, device):
    batch = []
    max_len = max([len(f) for f in formulas_list])
    for f in formulas_list:
        f_idx = [vocab.token_to_index[t] if t in vocab.token_to_index else vocab.token_to_index[vocab.unknown_token]
                 for t in f]
        eos = [vocab.token_to_index[vocab.eos_token]] * (max_len - len(f_idx) + 1)
        batch.append([vocab.token_to_index[vocab.sos_token]] + f_idx + eos)
    # we transpose here to make it compatible with LSTM input
    return torch.LongTensor(batch).T.contiguous().to(device)


def build_ordered_batches(formula_file, vocab, batch_size, device):
    formulas = []
    with open(formula_file) as f:
        for line in f:
            formulas.append(line.split())

    batches = []
    sorted_formulas = sorted(formulas, key=lambda x: len(x))
    for batch_ind in range((len(sorted_formulas) + batch_size - 1) // batch_size):
        batch_formulas = sorted_formulas[batch_ind * batch_size:(batch_ind + 1) * batch_size]
        batches.append(build_single_batch_from_formulas_list(batch_formulas, vocab, device))
    return batches
