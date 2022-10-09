# Ryszard Pulawski
import itertools
import math
import sys

import numpy as np
from numpy import linalg


# to sie na 99% da zoptymalizowac
def getStopienWielomianu(n, l):
    for d in range(1, l):
        suma = 1
        for j in range(1, d + 1):
            suma *= (n + j)
        suma /= math.factorial(d)
        if suma == l:
            return d
    return 0


N = int(sys.stdin.readline())
p = sys.stdin.readline().split()
p = [float(p[0])] + list(map(float, p[1:]))
n = len(p)

# to sa listy slownikow {wielowskaznik : wartosc}
wielowskaznikM = []
wielowskaznikL = []

for i in range(n):
    Li = sys.stdin.readline().split()
    Li = [float(Li[0])] + list(map(float, Li[1:]))
    Mi = sys.stdin.readline().split()
    Mi = [float(Mi[0])] + list(map(float, Mi[1:]))

    wielowskaznikLi = {}
    wielowskaznikMi = {}
    # zakladam, ze nie bedzie 2 takich samych wielomianow
    # ewentualnie ponizje do poprawy
    klucze = []
    nLi = len(Li)
    nMi = len(Mi)
    stopien_wielomianuL = getStopienWielomianu(n, nLi)
    stopien_wielomianuM = getStopienWielomianu(n, nMi)
    stopien_max = max(stopien_wielomianuL, stopien_wielomianuM)

    for m in range(stopien_max + 1):
        klucz = [i for i in itertools.product(range(stopien_max + 1), repeat=n) if sum(i) == m]
        klucz.reverse()
        for val in klucz:
            klucze.append(val)

    for j in range(max(nLi, nMi)):
        if j < nLi:
            wielowskaznikLi[tuple(klucze[j])] = Li[j]
        if j < nMi:
            wielowskaznikMi[tuple(klucze[j])] = Mi[j]

    wielowskaznikM.append(wielowskaznikMi)
    wielowskaznikL.append(wielowskaznikLi)

# -------------------------------------------------------- koniec wczytywania danych

# macierz pochodnych w punkcie, potem ja zapelnie
wielowskaznikDM = []
wielowskaznikDL = []

nL = len(wielowskaznikL)
nM = len(wielowskaznikM)

for i in range(max(nL, nM)):
    for j in range(n):
        if i < nL:
            wielowskaznikDLi = dict.fromkeys(wielowskaznikL[i].keys(), 0)
            for key, value in wielowskaznikL[i].items():
                if key[j] > 0:
                    newKey = list(key)
                    newKey[j] -= 1
                    newKey = tuple(newKey)
                    wielowskaznikDLi[newKey] += value * key[j]
            wielowskaznikDL.append(wielowskaznikDLi)

        if i < nM:
            wielowskaznikDMi = dict.fromkeys(wielowskaznikM[i].keys(), 0)
            for key, value in wielowskaznikM[i].items():
                if key[j] > 0:
                    newKey = list(key)
                    newKey[j] -= 1
                    newKey = tuple(newKey)
                    wielowskaznikDMi[newKey] += value * key[j]
            wielowskaznikDM.append(wielowskaznikDMi)

wielowskaznikDM = np.reshape(wielowskaznikDM, (n, n))
wielowskaznikDL = np.reshape(wielowskaznikDL, (n, n))


# w wielowskanikDL i wielowskaznikDM sa jacobiany dla licznika i mianownika
# teraz trzeba zlozyc z nich jacobian f i policzyc wartosc w punkcie p
def getValueInPoint(p, wielowskaznik):  # wyglada ok ta funkcja
    wynik = 0
    for key, value in wielowskaznik.items():
        if key == tuple(np.zeros(n)):
            wynik += value
            continue
        mnoznik = 1
        for k in range(len(key)):
            if key[k] == 0:
                continue
            mnoznik *= p[k] ** key[k]
        wynik += mnoznik * value
    return wynik


Df = []

for i in range(n):
    valL = getValueInPoint(p, wielowskaznikL[i])
    valM = getValueInPoint(p, wielowskaznikM[i])
    for j in range(n):
        valDL = getValueInPoint(p, wielowskaznikDL[i][j])
        valDM = getValueInPoint(p, wielowskaznikDM[i][j])
        # print(f"({valDL} * {valM} - {valL} * {valDM})/{valM}**2 = {(valDL * valM - valL * valDM) / valM ** 2}")
        Df.append((valDL * valM - valL * valDM) / valM ** 2)

Df = np.reshape(Df, (n, n))
# print(Df)
# teraz gdy mamy jacobian, obliczamy wartosci wlasne i wektory wlasne


vals, vects = linalg.eig(Df)

# print(vals, "<- vals   \n  vects->", vects)

wartoscWlasna = min(vals, key=abs)
# print("wartosc własna: ", wartoscWlasna)
mincol = list(vals).index(min(vals, key=abs))
# print("mincol", mincol, vects)
wektorWlasny = vects[:, mincol]
# print("wekt wlasny", wektorWlasny)
if wektorWlasny[0] < 0:
    wektorWlasny *= -1


# mamy wektory wlasne i wartosci wlasne, potrzebujemy zaimplementowac FAD


class Dzet:
    # n dlugosc listy
    # tu jeszcze chce jeszcze konstruktor kopiujacy
    def __init__(self, n):
        self.n = n
        self.wielomian = [0.] * n

    def __add__(self, other):
        nMax = max(self.n, other.n)
        wynik = Dzet(nMax)

        for i in range(nMax):
            if i < self.n:
                wynik.wielomian[i] += self.wielomian[i]
            if i < other.n:
                wynik.wielomian[i] += other.wielomian[i]

        return wynik

    def __mul__(self, other):
        wynik = Dzet(self.n + other.n - 1)
        for i in range(self.n):
            for j in range(other.n):
                wynik.wielomian[i + j] += self.wielomian[i] * other.wielomian[j]
        return wynik


dzety = []
# najpierw tworzymy dzety dla k = 2
# zapelniamy je wartosciami p i wektoraWlasnego
for i in range(n):
    dzety.append(Dzet(2))
    dzety[i].wielomian[0] = p[i]
    dzety[i].wielomian[1] = wektorWlasny[i]


# funkcja pomocnicza
def obliczFOdJ(wielowskaznik, dzety):
    wynik = Dzet(2)
    for key, value in wielowskaznik.items():
        if key == tuple(np.zeros(n)):
            wynik.wielomian[0] += value
            continue

        wynikTmp = Dzet(2)
        wynikTmp.wielomian[0] = 1

        keyTmp = list(key)
        while keyTmp != list(np.zeros(n)):
            for i in range(n):
                if keyTmp[i] != 0:
                    wynikTmp = wynikTmp * dzety[i]
                    keyTmp[i] -= 1

        tmp = Dzet(2)
        tmp.wielomian[0] = value

        wynikTmp = wynikTmp * tmp
        wynik = wynik + wynikTmp
    return wynik


def obliczWspolczynnikDzetLPrzezDzetM(dzetL, dzetM, k):
    wynik = [(dzetL.wielomian[0] / dzetM.wielomian[0])]

    for i in range(1, k + 1):
        tmp = dzetL.wielomian[i]
        for j in range(len(wynik)):
            if dzetM.n <= i - j:
                # print(f" {i} - {j} = {i-j}")
                continue
            tmp -= wynik[j] * dzetM.wielomian[i - j]
        wynik.append(tmp / dzetM.wielomian[0])

    return wynik[-1]


for k in range(2, N + 1):
    # tutaj musimy obliczyc f(dzety)
    JLessThanK = [0.] * n
    for i in range(n):
        dzetL = obliczFOdJ(wielowskaznikL[i], dzety)
        dzetM = obliczFOdJ(wielowskaznikM[i], dzety)
        print(dzetL.wielomian, " \n ", dzetM.wielomian)
        JLessThanK[i] = -obliczWspolczynnikDzetLPrzezDzetM(dzetL, dzetM, k)
        print(JLessThanK[i])

    Iden = np.identity(n) * (wartoscWlasna ** k)
    LS = np.subtract(Df, Iden)
    JK = np.linalg.solve(LS, JLessThanK)

    for i in range(n):
        dzety[i].wielomian.append(JK[i])
        dzety[i].n += 1

for dzet in dzety:
    for i in range(dzet.n):
        print("{:.17e}".format(dzet.wielomian[i]), end=" ")
    print()

# print na BACE

print("\nprint na bace")
# print "sdfsdfdsfs"
# for dzet in dzety:
#    for i in range(dzet.n):
#        print "{:.17e}".format(dzet.wielomian[i]),
#    print ""
