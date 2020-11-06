import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate formulas arguments')
    parser.add_argument('--source-file', type=str, default='results/formulas_test-2.txt',
                        help='Name of source file')
    parser.add_argument('--target-file', type=str, default='results/test.rec-2.txt',
                        help='Name of target file')
    args = parser.parse_args()
    with open(args.source_file) as source_f, open(args.target_file) as target_f:
        correct_formulas_count = 0
        incorrect_formulas_count = 0
        for l1, l2 in zip(source_f, target_f):
            l1 = l1.strip()
            l2 = l2.strip()
            if l1 == l2:
                correct_formulas_count += 1
            else:
                incorrect_formulas_count += 1
                print('Incorrect formula:\n\t%s - real\n\t%s - predicted' % (l1, l2))
        print('Correct formulas: %d\nIncorrect formulas: %d\nPercent of incorrect formulas: %.3f' % (
            correct_formulas_count,
            incorrect_formulas_count,
            100 * incorrect_formulas_count / (correct_formulas_count + incorrect_formulas_count)))


if __name__ == '__main__':
    main()
