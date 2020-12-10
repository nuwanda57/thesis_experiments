import torch


def build_single_batch_from_formulas_list(formulas_list, vocab, device):
    batch_in, batch_out = [], []
    max_len = max([len(f) for f in formulas_list])
    for f in formulas_list:
        f_idx = [vocab.token_to_index[t] if t in vocab.token_to_index else vocab.token_to_index[vocab.unknown_token]
                 for t in f]
        padding = [vocab.token_to_index[vocab.pad_token]] * (max_len - len(f_idx))
        batch_in.append([vocab.token_to_index[vocab.sos_token]] + f_idx + padding)
        batch_out.append(f_idx + [vocab.token_to_index[vocab.eos_token]] + padding)
    # we transpose here to make it compatible with LSTM input
    return torch.LongTensor(batch_in).T.contiguous().to(device), torch.LongTensor(batch_out).T.contiguous().to(device)


def build_ordered_batches(formula_file, vocab, batch_size, device):
    formulas = []
    with open(formula_file) as f:
        for line in f:
            formulas.append(line.split())

    batches = []
    order = range(len(formulas))  # This will be necessary for reconstruction
    sorted_formulas, order = zip(*sorted(zip(formulas, order), key=lambda x: len(x[0])))
    for batch_ind in range((len(sorted_formulas) + batch_size - 1) // batch_size):
        batch_formulas = sorted_formulas[batch_ind * batch_size:(batch_ind + 1) * batch_size]
        batches.append(build_single_batch_from_formulas_list(batch_formulas, vocab, device))
    return batches, order


def polish_to_standard(old_formula):
    if len(old_formula) == 0:
        raise 42
    if len(old_formula) == 1:
        return old_formula[0]
    symbol_count = 0
    num_count = 0
    reversed_f = old_formula[::-1][1:]
    i = 0
    for item in reversed_f:
        i += 1
        if item in ['^', '*', '+']:
            symbol_count += 1
        else:
            num_count += 1
        if symbol_count + 1 == num_count:
            break
    right = reversed_f[:i][::-1]
    left = reversed_f[i:][::-1]
    return polish_to_standard(left) + old_formula[-1] + polish_to_standard(right)


def polynom_to_normal_formula(polynom):
    polynom = polynom.split()
    formula = []
    number = None
    for s in polynom:
        if s == '<n>':
            if number is not None:
                formula.append(str(number))
            number = 0
        elif s in ['x', '^', '*', '+']:
            if number is not None:
                formula.append(str(number))
            number = None
            formula.append(s)
        else:
            if number is None:
                number = 0
            number *= 10
            number += int(s)
    if number is not None:
        formula.append(str(number))
    return polish_to_standard(formula)


def eval_polynom(polynom, x):
    formula = polynom_to_normal_formula(polynom)
    x_count = formula.count('x')
    formula = formula.replace('x', '%d')
    formula = formula.replace('^', '**')
    formula = formula % ((x,) * x_count)
    return eval(formula)



if __name__ == '__main__':
    print(polynom_to_normal_formula('<n> 1 3 <n> 1 2 x * + <n> 2 3 x <n> 2 ^ * + <n> 4 3 x <n> 3 ^ * +'))
    print(eval_polynom('<n> 1 3 <n> 1 2 x * + <n> 2 3 x <n> 2 ^ * + <n> 4 3 x <n> 3 ^ * +', 5))
