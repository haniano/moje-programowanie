# Hanna Nowak
import sys

import numpy as np
import math
from numpy import linalg
import math


def stopienD(n, maxEl):
    n = n + 1
    for d in range(1, maxEl):
        for i in range(1, d):
            n = n * (n + 1)

        wynik = n / math.factorial(d)

        if wynik >= maxEl:  # maybe rowne??
            return d
    return 0


def wielowskazniki(n, k):
    # moze wiecej przypadkow
    if not k: return [[0] * n]
    if not n: return []
    if n == 1:
        return [[k]]
    return [[0] + val for val in wielowskazniki(n - 1, k)] + \
           [[val[0] + 1] + val[1:] for val in wielowskazniki(n, k - 1)]


def bazaWielowskaznikow(n, d):
    baza = []

    for i in range(0, d + 1):
        baza.append(wielowskazniki(n, i)[::-1])  # [::-1] odwraca kolejnosc

    baza = [item for sublist in baza for item in sublist]  # flatten
    return baza


def zrobSlownik(t, n):
    slownik = {}
    d = stopienD(n, len(t))
    baza = bazaWielowskaznikow(n, d)

    for j in range(0, len(t)):
        slownik[tuple(baza[j])] = float(t[j])

    return slownik


def pochodnaDzielenia(L, Lp, M, Mp):
    wynik = (Lp * M - L * Mp) / (Mp ** 2)
    return wynik


def newTuple(tuple, index):
    newtuple = []
    for i in range(0, len(tuple)):
        if i == index and tuple[i] > 0:
            newtuple.append(tuple[i] - 1)
        else:
            newtuple.append(tuple[i])
    return newtuple


def wartoscFunkcjiPrim(funkcja, zmienna, p):
    wynik = 0
    for key, val in funkcja.items():  # dla kazdego elementu ze slownika bedziemy mnozyc wspolczynniki
        skladnik = val
        if val == 0:
            continue
        for i in range(0, len(key)):  # wyliczamy nowy wspolczynnik przed fragmentem
            # jesli jestesmy na zmiennej po ktorej liczymy pocodna
            if i == zmienna:
                if key[i] == 1:
                    continue
                elif p[i] == 0:
                    skladnik = 0
                else:
                    skladnik *= (key[i]) * (p[i] ** (key[i] - 1))  # wykladnik razy wartosc do (potegi -1)
            else:
                skladnik *= (p[i] ** key[i])
        wynik += skladnik
    return wynik


def wartoscFunkcji(funkcja, p):
    wynik = 0
    for key, val in funkcja.items():  # dla kazdego elementu ze slownika bedziemy mnozyc wspolczynniki
        skladnik = val
        for i in range(0, len(key)):  # wyliczamy nowy wspolczynnik przed fragmentem
            skladnik = skladnik * (p[i] ** key[i])
        wynik += skladnik
    return wynik


def wielomianRazyWielomian(funkcja1, funkcja2, ograniczenie):
    wynik = np.zeros(ograniczenie + 1)

    for i in range(0, len(funkcja1)):
        for j in range(0, len(funkcja2)):

            iloczynElementow = funkcja1[i] * funkcja2[j]

            if (i + j) < ograniczenie + 1:
                wynik[i + j] += iloczynElementow
    result = []
    for i in range(0, len(wynik)):
        result.append(wynik[i])

    return result


def wielomianDoPotegi(funkcja, stopien, ograniczenie):
    if stopien == 1:
        if len(funkcja) == ograniczenie + 1:
            return funkcja
        elif len(funkcja) > ograniczenie + 1:
            return funkcja[:ograniczenie + 1]
        else:
            while len(funkcja) != ograniczenie + 1:
                funkcja.append(0)
            return funkcja

    wynik = np.zeros(ograniczenie + 1)

    if stopien == 0:  # tego pewna nie jestem ale chyba whatever
        return wynik

    tempKrotka = []
    # tu przepisuje funkcje sobie zeby sie dalo duzo razy przemnazac
    for k in range(0, ograniczenie + 1):
        if k < len(funkcja):
            tempKrotka.append(funkcja[k])
        else:
            tempKrotka.append(0)

    for k in range(0, stopien - 1):
        if k > 0:  # bo pierwszy zalatwiony
            tempKrotka = []
            for el in wynik:
                tempKrotka.append(el)
        wynik = wielomianRazyWielomian(tempKrotka, funkcja, ograniczenie)

    result = []
    for i in range(0, len(wynik)):
        result.append(wynik[i])

    return result


def pierwszeNIEzero(tuple):
    for i in range(0, len(tuple)):
        if tuple[i] != 0:
            return i
    return 0


def FodJ(listaSlownikow, J, limit):
    wynik = np.zeros(limit + 1)

    for key, val in listaSlownikow.items():

        if val == 0:
            continue

        if all(k == 0 for k in key):
            wynik[0] += val
            continue
        else:
            index = pierwszeNIEzero(key)
            tempwynik = wielomianDoPotegi(J[index], key[index], limit)

            for i in range(index + 1, len(key)):

                if key[i] == 0:
                    continue
                else:
                    ttt = wielomianDoPotegi(J[i], key[i], limit)
                    tempwynik = wielomianRazyWielomian(ttt, tempwynik, limit)

            temptemphelp = []
            for i in range(0, len(tempwynik)):
                temptemphelp.append(tempwynik[i])

            for i in range(0, len(tempwynik)):
                temptemphelp[i] *= val

        # redukcja wyrazow podobmych
        for i in range(0, len(temptemphelp)):
            wynik[i] += temptemphelp[i]

    return wynik


def wspolczynnikIlorazu(licznik, mianownik, stopien):
    c = [licznik[0] / mianownik[0]]
    for i in range(1, stopien + 1):
        a = licznik[i]
        cb = 0
        for j in range(0, len(c)):
            if len(mianownik) > i - j:
                cb += c[j] * mianownik[i - j]
        c.append((a - cb) / mianownik[0])

    return c[stopien]


def lambaMin(wartWlasne):
    min = 0
    for i in range(0, len(wartWlasne)):
        if math.fabs(wartWlasne[i]) < wartWlasne[min]:
            min = i
    return min


N = int(sys.stdin.readline())
p = sys.stdin.readline().split()
p = [float(p[0])] + list(map(float, p[1:]))

n = len(p)

temp = []
listaSlownikowL = []
listaSlownikowM = []
slownik = {}

for i in range(0, n):
    t = sys.stdin.readline().split()
    t = [float(t[0])] + list(map(float, t[1:]))

    slownik = zrobSlownik(t, n)
    listaSlownikowL.append(slownik)

    t = sys.stdin.readline().split()
    t = [float(t[0])] + list(map(float, t[1:]))

    slownik = zrobSlownik(t, n)
    listaSlownikowM.append(slownik)

DF = np.zeros((n, n))

for i in range(0, n):
    for j in range(0, n):  # biore i-ty slownik i bede liczyc dla j-tej zmiennej
        Lp = wartoscFunkcjiPrim(listaSlownikowL[i], j, p)
        Mp = wartoscFunkcjiPrim(listaSlownikowM[i], j, p)
        L = wartoscFunkcji(listaSlownikowL[i], p)
        M = wartoscFunkcji(listaSlownikowM[i], p)

        DF[i][j] = (Lp * M - L * Mp) / (M * M)

eigen = np.linalg.eig(DF)

wartosciWlasne = eigen[0]
wektoryWlasne = eigen[1]

indexOfMin = lambaMin(wartosciWlasne)
lambdaMin = wartosciWlasne[indexOfMin]

J1 = wektoryWlasne[:, indexOfMin]

if J1[0] < 0:
    J1 *= -1
J = []

for i in range(0, n):
    J.append([])
    for j in range(0, N + 1):
        J[i].append(0)

for i in range(0, n):
    J[i][0] = p[i]
    J[i][1] = J1[i]

# teraz chce policzyc wszystkie pozostale J
# J jest wymiaru   liczba zmiannych(wymiar) X stopien ktory chce osiagnac
# czyli musze ja wywolac jeszcze stopien - 2 razy?
for i in range(2, N + 1):  # teraz bede liczyc od 2 stopnia J
    delta = []
    for j in range(0, n):  # musze to policzyc dla kazdej zmiennej
        iloczynLicznika = FodJ(listaSlownikowL[j], J, i)
        iloczynMianownika = FodJ(listaSlownikowM[j], J, i)

        wspolczynnikDzielenia = -wspolczynnikIlorazu(iloczynLicznika, iloczynMianownika, i)
        delta.append(wspolczynnikDzielenia)

    identycznosc = np.identity(n) * (lambdaMin ** i)
    DFsubIL = np.subtract(DF, identycznosc)
    Jtemp = np.linalg.solve(DFsubIL, delta)

    for j in range(0, n):
        J[j][i] += Jtemp[j]

for i in range(0, n):
    for j in range(0, N + 1):
        print("{:.17e}".format(J[i][j]), end=" ")
    print()
