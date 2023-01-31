def f(n):
    #for x in range(n):
    #    yield x
    #yield sum(range(n))
    yield from range(n)
    return sum(range(n))

a, b = yield from f(4)
