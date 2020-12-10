import argparse


def main(predicted_file, target_file):
    with open(predicted_file) as source_f, open(target_file) as target_f:
        correct_formulas_count = 0
        incorrect_formulas_count = 0
        for l1, l2 in zip(source_f, target_f):
            l1 = l1.strip()
            l2 = l2.strip()
            if l1 == l2:
                correct_formulas_count += 1
            else:
                incorrect_formulas_count += 1
                # print('Incorrect formula:\n\t%s - real\n\t%s - predicted' % (l1, l2))
        percent_correct = 100 * correct_formulas_count / (correct_formulas_count + incorrect_formulas_count)
        print('Correct formulas: %d\nIncorrect formulas: %d\nPercent of correct formulas: %.3f' % (
            correct_formulas_count,
            incorrect_formulas_count,
            percent_correct))
        return correct_formulas_count, incorrect_formulas_count, percent_correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate formulas arguments')
    parser.add_argument('--predicted-file', type=str, default='test.rec',
                        help='Name of source file')
    parser.add_argument('--target-file', type=str, default='formulas_test.txt',
                        help='Name of target file')
    args = parser.parse_args()
    main(args.predicted_file, args.target_file)
