import math
import random

import numpy as np


def sortColumn(matrix):
    matrix = np.matrix(matrix)
    print(matrix, end='\n\n')

    matrix = np.sort(matrix, axis=0)
    print(matrix)


matrix = [[2, 4, 3], [1, 3, 2], [7, 8, 6]]


# sortColumn(matrix)


import numpy as np


def matrixMerge(A, B):
    A = np.matrix(A)
    print(A, end='\n\n')

    B = np.matrix(B)
    print(B, end='\n\n')

    C1 = np.zeros((3, 8))
    C1[0:3, 0:4] = A
    C1[0:3, 4:8] = B

    print(C1, end='\n\n')
    print("Minimum kolejnych wierszy: ", C1.min(axis=1), end='\n\n')

    C2 = C1.reshape(6, 4)
    print(C2, end='\n\n')

    average = np.average(C2)

    print("Średnia całej macierzy: ", average, end='\n\n')

    indeces = np.argwhere(C2 > average)

    print("Indeksy elementów większych od średniej \n", indeces, end='\n\n')


# A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# B = [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]

A = np.random.randint(100, size=(3, 4))
B = np.random.randint(100, size=(3, 4))

matrixMerge(A, B)

import pandas as pd


def distIn10D(M):
    M = np.matrix(M)
    print(M, end='\n\n')

    squares = np.square(M)

    sums = squares.sum(axis=1)
    print(squares)
    print(sums)
    result = np.sqrt(sums)

    print(result)


M = np.random.randint(10, size=(100, 10))

# distIn10D(M)

from collections import Counter


import numpy as np

def najczestszaWartosc():
    M = np.random.randint(-10, 11, size=(10, 10))
    print(M, "\n\n")

    M[M < 0] = 0
    print(M, "\n\n")

    M[M % 2 != 0] = M[M % 2 != 0] * 2
    print(M, "\n\n")

    fr = [el for sublist in M for el in sublist]
    result = max(fr, key=fr.count)

    print("Najczęściej występująca wartość: ", result)


# najczestszaWartosc()

import pandas as pd
import numpy as np
import random
import math

def NA():
    M = np.ones((20, 5))

    print(M, "\n\n")

    for i in range(0, 15):
        x = random.randint(0, 19)
        y = random.randint(0, 4)
        M[x][y] = math.nan

    print(M, "\n\n")

    df = pd.DataFrame(M)
    df = df.dropna()

    print(df, "\n\n")


# NA()
