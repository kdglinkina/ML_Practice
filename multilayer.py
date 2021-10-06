import numpy as np


# Инициализируем пороговую функцию активации
def step_activation(x):
    return 0 if x < 0.5 else 1


# Инициализируем функцию обучения нейрона
def neural_training(color, material, price):
    x = np.array([color, material, price])

    # Указываем предпочтительные веса для 1 и 2 нейронов скрытого слоя
    hidden_w1 = [0.3, 0.3, 0]
    hidden_w2 = [0.4, -0.5, 1]

    # Упаковываем в матрицу и вектор связи
    weight1 = np.array([hidden_w1, hidden_w2])
    weight2 = np.array([-1, 1])

    # Вычисляем сумму на входах нейронов скрытого слоя
    sum_hidden = np.dot(weight1, x)
    print("Значения сумм на нейронах скрытого слоя: " + str(sum_hidden))

    # Проведя sum_hidden через функцию активации,
    # вычисляем выходные значения с каждого нейрона скрытого слоя
    out_hidden = np.array([step_activation(x) for x in sum_hidden])
    print("Значения на выходах нейронов скрытого слоя: " + str(out_hidden))

    # Вычисляем сумму выходного нейрона
    sum_end = np.dot(weight2, out_hidden)
    output = step_activation(sum_end)
    print("Выходное значение: " + str(output))

    return output


color = 1
material = 0
price = 1

res = neural_training(color, material, price)
if res == 1:
    print("Покупаю!")
else:
    print("Не сегодня")
