import formulas_vae_v4.formula_config as my_formula_config

import torch


def build_single_batch_from_formulas_list(formulas_list, device):
    batch_in, batch_out = [], []
    max_len = max([len(f) for f in formulas_list])
    for f in formulas_list:
        f_idx = [my_formula_config.TOKEN_TO_INDEX[t] for t in f]
        padding = [my_formula_config.TOKEN_TO_INDEX[my_formula_config.PADDING]] * (max_len - len(f_idx))
        batch_in.append([my_formula_config.TOKEN_TO_INDEX[my_formula_config.START_OF_SEQUENCE]] + f_idx + padding)
        batch_out.append(f_idx + [my_formula_config.TOKEN_TO_INDEX[my_formula_config.END_OF_SEQUENCE]] + padding)
    # we transpose here to make it compatible with LSTM input
    return torch.LongTensor(batch_in).T.contiguous().to(device), torch.LongTensor(batch_out).T.contiguous().to(device)


def build_ordered_batches(formula_file, batch_size, device):
    formulas = []
    with open(formula_file) as f:
        for line in f:
            formulas.append(line.split())

    batches = []
    order = range(len(formulas))  # This will be necessary for reconstruction
    sorted_formulas, order = zip(*sorted(zip(formulas, order), key=lambda x: len(x[0])))
    for batch_ind in range((len(sorted_formulas) + batch_size - 1) // batch_size):
        batch_formulas = sorted_formulas[batch_ind * batch_size:(batch_ind + 1) * batch_size]
        batches.append(build_single_batch_from_formulas_list(batch_formulas, device))
    return batches, order
