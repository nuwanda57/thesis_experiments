import formulas_vae_v4.formula_config as my_formula_config

from collections import deque


def maybe_get_valid(polish_formula):
    numbers_required = 1
    valid_polish_formula = []
    for token in polish_formula:
        if token in {my_formula_config.START_OF_SEQUENCE, my_formula_config.END_OF_SEQUENCE,
                     my_formula_config.PADDING, my_formula_config.NUMBER_SYMBOL}:
            continue
        if token in my_formula_config.OPERATORS:
            valid_polish_formula.append(token)
            numbers_required += (my_formula_config.OPERATORS[token].arity - 1)
        else:
            valid_polish_formula.append(token)
            numbers_required -= 1
            if numbers_required == 0:
                return valid_polish_formula
    return None


def get_formula_representation(valid_polish_formula_):
    valid_polish_formula = valid_polish_formula_
    if isinstance(valid_polish_formula, str):
        valid_polish_formula = valid_polish_formula.split()
    if len(valid_polish_formula) == 0:
        return ''
    stack = deque(valid_polish_formula)
    args = deque()
    while len(stack) != 0:
        token = stack.pop()
        if token in my_formula_config.OPERATORS:
            operator = my_formula_config.OPERATORS[token]
            params = [args.popleft() for _ in range(operator.arity)]
            args.appendleft(operator.repr(params))
        else:
            args.appendleft(token)

    assert len(args) == 1, f"{args}, {valid_polish_formula}"
    return args.pop()


if __name__ == '__main__':
    assert get_formula_representation(['mult', 'add', '1', 'x', 'add', 'sin', '5', 'cos', '6']) \
           == '((1) + (x)) * ((np.sin(5)) + (np.cos(6)))'
    assert get_formula_representation([]) == ''
    assert get_formula_representation(['1']) == '1'
    assert get_formula_representation(['x']) == 'x'

    assert maybe_get_valid([]) is None
    assert maybe_get_valid(['mult']) is None
    assert maybe_get_valid(['x']) == ['x']
    assert maybe_get_valid(['x', '3']) == ['x']
    assert maybe_get_valid(['mult', 'x', '1', '4']) == ['mult', 'x', '1']
    assert maybe_get_valid(['mult', 'x', '1', 'add']) == ['mult', 'x', '1']
    assert maybe_get_valid(['mult', 'x', 'add', '4', '3']) == ['mult', 'x', 'add', '4', '3']
    assert maybe_get_valid(['mult', 'add', 'sin', '5', 'cos', '6', 'add', '4', '3', 'add', '3', '1']) \
           == ['mult', 'add', 'sin', '5', 'cos', '6', 'add', '4', '3']

    print(get_formula_representation(['mult', 'add', '1', 'x', 'add', 'sin', '5', 'cos', '6']))
    print(get_formula_representation('sin sin sin x'.split()))

    print(get_formula_representation('mult mult x x mult x x'.split()))
    print(get_formula_representation('mult mult x x mult x x'.split()))
    print(get_formula_representation('mult mult x x mult x x'.split()))
    print(get_formula_representation('mult mult x x mult x x'.split()))
    print(get_formula_representation('mult mult x x mult x x'.split()))
    print(get_formula_representation('mult mult x x mult x x'.split()))
