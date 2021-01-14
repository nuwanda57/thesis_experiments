import numpy as np

import formulas_vae_v2.formula_config as my_formula_config


def generate_polynomials_dataset_from_vocab(count, vocab=my_formula_config.COEFFICIENTS, max_power=7):
    """
    :param count: number of formulas to generate
    :param vocab: list of all available coefficients
    :return: list of formulas
    """
    def build_polynomial(coefs):
        if len(coefs) <= 0:
            raise 42
        zero_term = coefs[0]
        if len(coefs) == 1:
            return zero_term
        first_term = f'{coefs[1]} x mult add'
        if len(coefs) == 2:
            return f'{zero_term} {first_term}'
        standard_term_template = '%s x %d pow mult add'
        standard_polynomial_part = ' '.join(
            [standard_term_template % (coef, i + 2) for i, coef in enumerate(coefs[2:])])
        return f'{zero_term} {first_term} {standard_polynomial_part}'

    coefs = [np.random.choice(vocab, size=(count // (max_power + 1), power)) for power in range(1, max_power + 2)]

    res = []
    for c in coefs:
        for i in c:
            res.append(build_polynomial(i))
    return res


def generate_formulas_dataset(filenames, counts, type='polynomial_vocab', **kwargs):
    if type == 'polynomial_vocab':
        while True:
            formulas = generate_polynomials_dataset_from_vocab(np.sum(counts) * 10, **kwargs)
            formulas = np.unique(formulas)
            if len(formulas) >= np.sum(counts):
                formulas = formulas[:np.sum(counts)]
                break
    else:
        raise 42

    np.random.shuffle(formulas)
    split_formulas = np.split(formulas, [sum(counts[:i]) for i in range(1, len(counts))])

    for formulas_bucket, filename in zip(split_formulas, filenames):
        with open(filename, 'w') as f:
            f.write('\n'.join(formulas_bucket))


def main(filenames, counts, max_power):
    generate_formulas_dataset(filenames, counts, max_power=max_power)


if __name__ == '__main__':
    main(['train.txt', 'val.txt', 'test.txt'], [20, 10, 10], 7)
