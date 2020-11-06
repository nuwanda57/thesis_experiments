TEST_FILE = 'formulas_test-2.txt'
RESULT_FILE = 'test.rec-2.txt'
ANALYSIS_FILE = 'analysis.txt'


def main():
    with open(TEST_FILE) as test_f, open(RESULT_FILE) as res_f:
        correct_formulas_count = 0
        incorrect_formulas_count = 0
        for l1, l2 in zip(test_f, res_f):
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
