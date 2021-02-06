from formulas_vae_v4 import formula_config as my_formula_config
from formulas_vae_v4 import formula_utils as my_formula_utils


import numpy as np


def evaluate(formula, xs):
    formula = formula.replace('x', 'xs')
    return eval(formula)


def evaluate_file(filename, xs):
    formulas = []
    with open(filename) as f:
        for line in f:
            formulas.append(my_formula_utils.get_formula_representation(line.strip()))
    results = []
    for formula in formulas:
        results.append(evaluate(formula, xs))
    return results


if __name__ == '__main__':
    xs = np.linspace(0.0, 3.0, num=100)
    print(evaluate('((1) + (x)) * ((np.sin(5)) + (np.cos(6)))', xs))
