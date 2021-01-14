import formulas_vae_v2.formula_config as my_formula_config


def split_numbers_into_tokens(formula):
    processed_formula = []
    for sym in formula:
        if sym in my_formula_config.OPERATORS or sym in my_formula_config.VARIABLES:
            processed_formula.append(sym)
            continue
        processed_formula.append(my_formula_config.NUMBER_START_SYMBOL)
        processed_formula += [token for token in sym]
    return processed_formula


def unify_tokens_into_numbers(formula):

    def maybe_add_number(n, f):
        if len(n) > 0:
            f.append(n)
            return ''
        return n

    processed_formula = []
    cur_number = ''

    for sym in formula:
        if sym in my_formula_config.OPERATORS or sym in my_formula_config.VARIABLES:
            cur_number = maybe_add_number(cur_number, processed_formula)
            processed_formula.append(sym)
            continue
        if sym == my_formula_config.NUMBER_START_SYMBOL:
            cur_number = maybe_add_number(cur_number, processed_formula)
            continue
        cur_number += sym

    _ = maybe_add_number(cur_number, processed_formula)
    return processed_formula


if __name__ == '__main__':
    print(split_numbers_into_tokens('10 13 x mult add 11 x 2 pow mul add 35 x 3 pow mul add 28 x 4 pow mul add'.split()))
    print(unify_tokens_into_numbers('<n> 1 x mult <n> 1 3 x <n> 3 pow mult add'.split()))
