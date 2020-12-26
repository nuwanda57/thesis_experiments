def main():
    with open('./../ff15.txt') as f:
        lines_test = f.readlines()
    with open('./../ff16.txt') as f:
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
