import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np


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


def eval_polynom(polynom, xs):
    try:
        formula = polynom_to_normal_formula(polynom)
    except:
        print('err', end=' ')
        return None
    x_count = formula.count('x')
    formula = formula.replace('*x^', '*%f^')
    formula = formula.replace('*x+', '*%f+')
    formula = formula.replace('^', '**')
    if formula[-2:] == '*x':
        formula = formula[:-1] + '%f'
    if 'x' in formula:
        # print(formula)
        print('err', end=' ')
        return None

    results = []
    for x in xs:
        try:
            results.append(eval(formula % ((x,) * x_count)))
        except:
            print('err', end=' ')
            return None

    results = [eval(formula % ((x,) * x_count)) for x in xs]
    return results


def reconstruction_mses(rec_file, test_file, xs):
    results = []
    with open(rec_file) as rec, open(test_file) as test:
        for rec_line, test_line in zip(rec, test):
            rec_results, test_results = eval_polynom(rec_line, xs), eval_polynom(test_line, xs)
            if rec_results is None:
                results.append(len(xs) * 100)
                continue
            if test_results is None:
                print(test_line)
                raise 42
            results.append(mean_squared_error(rec_results, test_results))
    print('\nfinished processing %s' % rec_file)
    return results


def mean_reconstruction_mse(rec_file, test_file, xs):
    mses = reconstruction_mses(rec_file, test_file, xs)
    return np.mean(mses)


def reconstruction_metric_eval(rec_file, test_file, xs, metric):
    results = []
    with open(rec_file) as rec, open(test_file) as test:
        for rec_line, test_line in zip(rec, test):
            rec_results, test_results = eval_polynom(rec_line, xs), eval_polynom(test_line, xs)
            if rec_results is None:
                results.append(len(xs) * 100)
                continue
            if test_results is None:
                print(test_line)
                raise 42
            if metric == 'mse':
                results.append(mean_squared_error(rec_results, test_results))
            elif metric == 'mae':
                results.append(mean_absolute_error(rec_results, test_results))
            elif metric == 'male':
                rec_results = np.array(rec_results)
                test_results = np.array(test_results)
                results.append(np.mean(np.abs(np.log(1 + np.abs(test_results)) - np.log(1 + np.abs(rec_results)))))
            else:
                raise 42
    print('\nfinished processing %s' % rec_file)
    return results


def mean_reconstruction_metric_eval(rec_file, test_file, xs, metric):
    metric_values = reconstruction_metric_eval(rec_file, test_file, xs, metric)
    return np.mean(metric_values)


if __name__ == '__main__':
    print(polynom_to_normal_formula('<n> 1 3 <n> 1 2 x * + <n> 2 3 x <n> 2 ^ * + <n> 4 3 x <n> 3 ^ * +'))
    print(eval_polynom('<n> 1 <n> 2 x * +', [5, 1, 2]))
    print(eval_polynom('<n> 1 5 <n> 1 9 x * + <n> 7 x <n> 2 ^ * + <n> 4 5 x <n> 3 ^ * +', [5,1,2]))
    # print(sorted(reconstruction_mses('./../../rec_100', './../../formulas_test_5_10.txt', [0.3, 0.5, 0.2, 0.1])))
    # print(mean_reconstruction_mse('./../tt2', './../tt1.txt', [0.3, 0.5, 0.2, 0.1]))
    # print(reconstruction_metric_eval('./../rec_400', './../tt1.txt', list(np.linspace(0.1, 1, 25)), 'mae'))
