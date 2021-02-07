from formulas_vae_v4 import formula_config as my_formula_config
from formulas_vae_v4 import formula_utils as my_formula_utils


import numpy as np
import numbers


def evaluate(formula, xs):
    formula = formula.replace('x', 'xs')
    res = eval(formula)
    if isinstance(res, numbers.Number):
        res = [res] * len(xs)
    return res


def evaluate_file(filename, xs):
    formulas = []
    with open(filename) as f:
        for line in f:
            formulas.append(my_formula_utils.get_formula_representation(line.strip().split()))
    results = []
    for formula in formulas:
        results.append(evaluate(formula, xs))
    return results


if __name__ == '__main__':
    xs = np.linspace(0.0, 2.0, num=100)
    print(evaluate('((1) + (x)) * ((np.sin(5)) + (np.cos(6)))', xs))
    print(evaluate('(((np.sin(((np.cos(2)) + (x)) - ((1) * (np.cos(2))))) - (np.sin(3))) + (np.sin(3))) * (x)', xs))
    print(evaluate('x * np.sin(x)', xs))
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(
        evaluate('x * np.sin(x)', xs),
        evaluate('(((np.sin(((np.cos(2)) + (x)) - ((1) * (np.cos(2))))) - (np.sin(3))) + (np.sin(3))) * (x)', xs)))
