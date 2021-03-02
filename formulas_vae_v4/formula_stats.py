import formulas_vae_v4.formula_config as formula_config
from copy import deepcopy


def get_all_postfix(nodes_cnt, tokens, postfix_for_count_and_vars_pair, vars_tokens, vars_to_add=1):
    if (nodes_cnt, vars_to_add) in postfix_for_count_and_vars_pair:
        return postfix_for_count_and_vars_pair[(nodes_cnt, vars_to_add)]
    if nodes_cnt == 0:
        if vars_to_add != 0:
            raise 42
        return []
    if nodes_cnt == 1:
        if vars_to_add == 1:
            return vars_tokens
        return []
    formulas = []
    for tok in tokens:
        arity = 0
        if tok in formula_config.OPERATORS:
            arity = formula_config.OPERATORS[tok].arity
        if vars_to_add + arity - 1 <= 0:
            continue
        postfixes = get_all_postfix(nodes_cnt - 1, tokens, postfix_for_count_and_vars_pair, vars_tokens,
                                    vars_to_add + arity - 1)
        formulas += [[tok] + p for p in postfixes]
    postfix_for_count_and_vars_pair[(nodes_cnt, vars_to_add)] = formulas
    return formulas


def get_all_formulas(nodes_cnt, tokens):
    vars_tokens = []
    for tok in tokens:
        if tok not in formula_config.OPERATORS:
            vars_tokens.append([tok])
    postfix_for_count_and_vars_pair = dict()
    formulas = get_all_postfix(nodes_cnt, tokens, postfix_for_count_and_vars_pair, vars_tokens, 1)
    print(len(formulas))
    print(formulas[:50])


if __name__ == '__main__':
    tokens = list(set(formula_config.INDEX_TO_TOKEN) - set(formula_config.SERVICE_TOKENS))
    get_all_formulas(12, tokens)
