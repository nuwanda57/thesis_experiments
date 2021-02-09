import formulas_vae_v4.formula_config as my_formula_config

from collections import deque


def clear_redundant_operations(polish_formula_prefix):
    tail_number_count = 0
    while tail_number_count < len(polish_formula_prefix) and \
            polish_formula_prefix[len(polish_formula_prefix) - 1 - tail_number_count] == my_formula_config.NUMBER_SYMBOL:
        tail_number_count += 1

    if tail_number_count == 0:
        return

    if tail_number_count < len(polish_formula_prefix) and \
            polish_formula_prefix[len(polish_formula_prefix) - 1 - tail_number_count] in my_formula_config.OPERATORS:
        operator_name = polish_formula_prefix[len(polish_formula_prefix) - 1 - tail_number_count]
        if my_formula_config.OPERATORS[operator_name].arity == tail_number_count:
            for _ in range(tail_number_count + 1):
                polish_formula_prefix.pop()
            polish_formula_prefix.append(my_formula_config.NUMBER_SYMBOL)
            clear_redundant_operations(polish_formula_prefix)


def maybe_get_valid(polish_formula):
    numbers_required = 1
    valid_polish_formula = []
    for token in polish_formula:
        if token in {my_formula_config.START_OF_SEQUENCE, my_formula_config.END_OF_SEQUENCE,
                     my_formula_config.PADDING}:
            continue
        if token in my_formula_config.OPERATORS:
            valid_polish_formula.append(token)
            numbers_required += (my_formula_config.OPERATORS[token].arity - 1)
        else:
            valid_polish_formula.append(token)
            clear_redundant_operations(valid_polish_formula)
            numbers_required -= 1
            if numbers_required == 0:
                return valid_polish_formula
    return None


def get_formula_representation(valid_polish_formula):
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

    assert maybe_get_valid(['<n>']) == ['<n>']
    assert maybe_get_valid(['<n>', '3']) == ['<n>']
    assert maybe_get_valid(['mult', '<n>', '<n>', '<n>']) == ['<n>']
    assert maybe_get_valid(['sin', '<n>']) == ['<n>']
    assert maybe_get_valid(['sin', 'cos', 'add', '<n>', '<n>']) == ['<n>']
    assert maybe_get_valid(['mult', 'add', 'sin', '<n>', 'cos', '6', 'add', '4', 'add', '<n>', '<n>']) \
           == ['mult', 'add', '<n>', 'cos', '6', 'add', '4', '<n>']

