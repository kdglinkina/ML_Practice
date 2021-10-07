import numpy as np


# функция активация - гиперболический тангенс
def f(x):
    return 2 / (1 + np.exp(-x)) - 1


# функция вычисления производной
def df(x):
    return 0.5 * (1 + x) * (1 - x)


# Инициализируем веса 1го и 2го слоя
weight1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
weight2 = np.array([0.2, 0.3])


# Пропускаем вектор наблюдений через нейронную сеть
def go_neural(inp):
    sum = np.dot(weight1, inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(weight2, out)
    y = f(sum)
    return (y, out)


def train(epoch):
    global weight2, weight1
    lmd = 0.01  # шаг обучения
    N = 10000  # число итераций при обучении
    count = len(epoch) # размер обучающей выборки
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # случайных выбор входного сигнала из обучающей выборки
        y, out = go_neural(x[0:3])  # прямой проход по НС и вычисление выходных значений нейронов
        e = y - x[-1]  # ошибка = выход сети Y вычесть желаемое значение
        delta = e * df(y)  # локальный градиент
        weight2[0] = weight2[0] - lmd * delta * out[0]  # корректировка веса первой связи
        weight2[1] = weight2[1] - lmd * delta * out[1]  # корректировка веса второй связи

        delta2 = weight2 * delta * df(out)  # вектор из 2-х величин локальных градиентов

        # корректировка связей нейронов первого слоя
        weight1[0, :] = weight1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        weight1[1, :] = weight1[1, :] - np.array(x[0:3]) * delta2[1] * lmd


# обучающая выборка, где последнее число - ожидаемое значение
epoch = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]

train(epoch)  # запуск обучения сети

# проверка полученных результатов
for x in epoch:
    y, out = go_neural(x[0:3])
    print(f"Выходное значение НС: {y} => {x[-1]}")
