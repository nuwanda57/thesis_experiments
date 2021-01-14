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
                stack.append(token)
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


if __name__ == '__main__':
    print(evaluate('3 x mult', [2]))
