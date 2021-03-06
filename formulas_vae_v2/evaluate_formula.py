from formulas_vae_v2 import formula_config as my_formula_config


class Error(Exception):
    pass


class UnknownOperatorError(Error):
    pass


class UnknownTokenError(Error):
    pass


def apply_operator(operator, params):
    if operator == 'add':
        return float(params[0]) + float(params[1])
    if operator == 'mult':
        return float(params[0]) * float(params[1])
    if operator == 'pow':
        return float(params[0]) ** float(params[1])
    raise UnknownOperatorError


def evaluate(formula, xs):
    formula = formula.split()
    results = []
    stack = list()
    for x in xs:
        for token in formula:
            if token in my_formula_config.VARIABLES:
                stack.append(x)
                continue
            if token in my_formula_config.COEFFICIENTS:
                stack.append(int(token))
                continue
            if token in my_formula_config.OPERATORS:
                params_count = my_formula_config.OPERATORS[token]
                assert len(stack) >= params_count, (
                    f'Not enought parameters in stack. Operator {token} requires {params_count} params')
                params = [stack.pop() for i in range(params_count)]
                stack.append(apply_operator(token, params))
            else:
                raise UnknownTokenError(token)
        assert len(stack) == 1, f'Length of the stack must be one after the evaluation. It\'s {len(stack)}'
        results.append(stack.pop())
    return results


def evaluate_file(filename, xs):
    formulas = []
    with open(filename) as f:
        for line in f:
            formulas.append(line.strip())
    results = []
    for formula in formulas:
        try:
            results.append(evaluate(formula, xs))
        except:
            results.append(None)
    return results


if __name__ == '__main__':
    import numpy as np
    from sklearn.metrics import mean_squared_error
    # print(evaluate('3 x mult', [2]))
    xs = np.linspace(0.0, 3.0, num=100)
    a = evaluate('0 x x mult add x x mult add 2 x mult add 3 x mult add 2 x mult add 1 x mult add 4 x 4 pow mult add 2 x 4 pow mult add', xs)
    print(a)
    print(np.log(mean_squared_error(29 * xs + 4 * xs ** 4, a)))
    # print(evaluate(["3","1","x","2","x","pow","pow","mult","add","2","x","pow","pow","mult","add"], [0.5]))
