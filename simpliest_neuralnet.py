# Простейшая нейросеть: однослойный перцептрон, true\false
# Инициализация входных параметров:
#   Кастомизированные для 3х элементов входные данные, веса, пороговое значение.

x_inputs = [-0.1, 0.5, 0.2]
w_weights = [0.4, 0.3, 0.6]
threshold = 0


# Инициализация пороговой функции активации:
def step_activation(weighted_sum):
    if weighted_sum > threshold:
        return 1
    else:
        return 0


# Инициализация функции перцептрона
def perceptron():
    weighted_sum = 0
    for x, w in zip(x_inputs, w_weights):
        weighted_sum += x * w  # Взвешенная сумма входов нейрона
    return step_activation(weighted_sum)


# Вывод результата
output = perceptron()
print("Output: " + str(output))
