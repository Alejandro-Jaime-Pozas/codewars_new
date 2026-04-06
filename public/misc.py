def countup(n):
    for i in range(n):
        yield i

gen = countup(4)
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
