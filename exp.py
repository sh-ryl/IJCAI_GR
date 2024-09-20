e = [1, 3, 2]

d = {
    3: 4,
    1: 6,
    2: 5
}

d = enumerate(d)
print(d)

d = sorted(d, key=lambda item: item[1])
print(d)
