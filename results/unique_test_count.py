def main():
    with open('./../formulas_test_5_10.txt') as f:
        lines_test = f.readlines()
    with open('./../formulas_train_5_10.txt') as f:
        lines_train = set(f.readlines())

    ans = 0
    k = set()
    for line in lines_test:
        if line not in lines_train:
            ans += 1
            k.add(line)
    print(ans)
    print(len(k))


if __name__ == '__main__':
    main()
