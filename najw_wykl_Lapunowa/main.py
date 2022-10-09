import math


# zla pochodna inna powinna byc wielu zmiennych
def pochodnaFunkcji(xk):
    v = [-2.8 * xk[0] + 1, 0.3]
    return v


def wartoscFunkcji(xk):
    v = [1 - 1.4 * xk[0] ** 2 + xk[1], 0.3 * xk[0]]
    return v


def wyklaLap(x0, v0, normv0, n):
    xk = x0
    vk = v0
    lambdy = []
    vfalkak = [0, 0]
    vkOk = [0, 0]
    for k in range(0, n):
        xktemp = xk
        xk = wartoscFunkcji(xktemp)
        Ak = pochodnaFunkcji(xktemp)
        vfalkak[0] = Ak[0] * vk[0]
        vfalkak[1] = Ak[1] * vk[1]
        vkOk[0] = vfalkak[0] / math.sqrt(vfalkak[0] ** 2 + vfalkak[1] ** 2)
        vkOk[1] = vfalkak[1] / math.sqrt(vfalkak[0] ** 2 + vfalkak[1] ** 2)
        lambdy.append(math.sqrt(vkOk[0] ** 2 + vkOk[1] ** 2))

    wynik = 0
    for i in range(0, len(lambdy)):
        wynik += math.log(lambdy[i], 10)

    return wynik / n


print(wyklaLap([1, 0], [1, 1], 1, 10000))
