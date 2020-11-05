import numpy as np

import argparse


def generate_polynomials_dataset(count, float_precision=1e3, max_coef=1e3, max_power=5):
    def build_polynomial(coefs):
        max_power = len(coefs)
        # print(coefs)
        # print(e)
        if max_power < 0:
            raise 42
        zero_term = '%.3f' % coefs[0]
        if max_power == 0:
            return zero_term
        first_term = '%.3f x * +' % coefs[1]
        if max_power == 1:
            return '%s %s' % (zero_term, first_term)
        standard_term_template = '%.3f x %d ^ * +'
        standard_polynomial_part = ' '.join([standard_term_template % (c, i + 2) for i, c in enumerate(coefs[2:])])
        return '%s %s %s' % (zero_term, first_term, standard_polynomial_part)

    coefs = np.random.randint(
        -max_coef * float_precision,
        max_coef * float_precision,
        size=(count, max_power + 1)
    ) / float_precision

    return np.apply_along_axis(build_polynomial, 1, coefs)


def generate_formulas_dataset(filenames, counts, type='polynomial', **kwargs):
    if type == 'polynomial':
        print(np.sum(counts))
        formulas = generate_polynomials_dataset(np.sum(counts), **kwargs)
    else:
        raise 42

    np.random.shuffle(formulas)
    split_formulas = np.split(formulas, [sum(counts[:i]) for i in range(1, len(counts))])
    print(len(split_formulas[0]))
    print(len(formulas))

    for formulas_bucket, filename in zip(split_formulas, filenames):
        with open(filename, 'w') as f:
            f.write('\n'.join(formulas_bucket))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate formulas arguments')
    parser.add_argument('--filenames', type=str, nargs='+',
                        help='Names of the files in which formulas must be saved')
    parser.add_argument('--counts', type=int, nargs='+',
                        help='Amounts of formulas that must be saved to each of the files')
    parser.add_argument('--max-power', type=int, required=False, default=5,
                        help='If polynomials, max power of a polynomial')

    args = parser.parse_args()
    generate_formulas_dataset(args.filenames, args.counts, max_power=args.max_power)
