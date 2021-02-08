from formulas_vae_v4 import formula_config as my_formula_config
from formulas_vae_v4 import formula_utils as my_formula_utils

import numpy as np
from copy import copy
import numbers
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def _evaluate_with_coeffs(formula, xs, ys):

    def optimization_func(coefs):
        optimized_formula = copy(formula)
        for c in coefs:
            optimized_formula = optimized_formula.replace(my_formula_config.NUMBER_SYMBOL, str(c), 1)
        res = eval(optimized_formula)
        if isinstance(res, numbers.Number):
            res = [res] * len(xs)
        return mean_squared_error(res, ys)

    coefs_cnt = formula.count(my_formula_config.NUMBER_SYMBOL)
    coefs = [0] * coefs_cnt

    res_minimize = minimize(optimization_func, coefs)
    mse, coefs = res_minimize.fun, res_minimize.x

    optimized_formula = copy(formula)
    for c in coefs:
        optimized_formula = optimized_formula.replace(my_formula_config.NUMBER_SYMBOL, str(c), 1)

    res = eval(optimized_formula)
    if isinstance(res, numbers.Number):
        res = [res] * len(xs)

    return mse, res, coefs, optimized_formula


def evaluate(formula, xs, ys, handle_coefs=True):
    formula = formula.replace('x', 'xs')
    mse, res, coefs, optimized_formula = None, None, None, copy(formula)
    if handle_coefs and my_formula_config.NUMBER_SYMBOL in formula:
        mse, res, coefs, optimized_formula = _evaluate_with_coeffs(formula, xs, ys)
    else:
        res = eval(formula)
        if isinstance(res, numbers.Number):
            res = [res] * len(xs)
        mse = mean_squared_error(res, ys)
    return mse, res, coefs, optimized_formula


def evaluate_file(filename, xs, ys):
    formulas = []
    with open(filename) as f:
        for line in f:
            formulas.append(my_formula_utils.get_formula_representation(line.strip().split()))
    results = []
    for formula in formulas:
        results.append(evaluate(formula, xs, ys))
    return results


if __name__ == '__main__':
    from math import isclose
    xs = np.linspace(0.0, 2.0, num=10)

    name = 'test 1'
    formula = 'x * np.sin(<n>)'
    ys = xs * np.sin(5)
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 2'
    formula = 'x * np.sin(<n>) + 2'
    ys = xs * np.sin(5) + 2
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 3'
    formula = 'x * np.sin(<n>) + <n>'
    ys = xs * np.sin(5) + 150
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 4'
    formula = 'x * np.sin(<n> + np.cos(<n>)) + 150 * <n>'
    ys = xs * np.sin(5 + np.cos(2)) + 300
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 5'
    formula = '5'
    ys = [5] * len(xs)
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 6'
    formula = '<n>'
    ys = [-12312] * len(xs)
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 7'
    formula = 'x * x * np.cos(x) + x * x * <n> + <n>'
    ys = xs * xs * np.cos(xs) + xs * xs * 23 + 25
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

    name = 'test 8'
    formula = 'x * x * x * <n> + x * x * <n> + <n> * x + <n>'
    ys = xs * xs * xs * 2.56 + xs * xs * 0.57 + xs * 100 + 1000
    mse, res, coefs, optimized_formula = evaluate(formula, xs, ys)
    assert isclose(
        mse,
        0, abs_tol=1e-5)
    print(f'{name}: real formula {formula}\n\tmse: {mse}\n\toptimized_formula: {optimized_formula}\n')

