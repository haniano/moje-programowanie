import numpy as np
import math


# punkt stały - p = (p1,...pn)
# wymiar - n
# wartosc wlasna - lambda
# wektor wlasny - u
# stopien rozwiniecia algorytmu - N


def ileJednomianow(n, d):
    licznik = 1
    n = n + 1
    while licznik != d:
        licznik = licznik + 1
        n = n * (n + 1)

    return n / math.factorial(d)


N = input()
p = input()
punkyStaly = []
punktStaly = p.strip().split(" ")

n = len(punktStaly)
k = 2 * n + 2
M = []
L = []

print(k)

coef = np.zeros[k, n]

for i in range(0, n):
    L.append(input())
    M.append(input())


print(L)
print(M)
