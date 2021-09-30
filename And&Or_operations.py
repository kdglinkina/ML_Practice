# Инициализация входных значений
x1_input = [0, 1, 0, 1]
x2_input = [0, 0, 1, 1]
w_weights = [1, 1, 1, 1]
threshold = 1


# Инициализация порога логической функции AND
def AND(weighted_sum):
    if weighted_sum <= threshold:
        return 0
    elif weighted_sum > threshold:
        return 1


# Инициализация порога логической функции OR
def OR(weighted_sum):
    if weighted_sum < threshold:
        return 0
    elif weighted_sum >= threshold:
        return 1


# Инициализация функции перцептрона для AND и OR
def perceptron():
    resultOR = [0, 0, 0, 0]
    resultAND = [0, 0, 0, 0]
    i = 0
    for x1, x2, w in zip(x1_input, x2_input, w_weights):
        weighted_sum = x1 * w + x2 * w  # Взвешенная сумма входов нейрона
        resultOR[i] = OR(weighted_sum)
        resultAND[i] = AND(weighted_sum)
        i += 1
    print("Для массивов")
    print("X1:" + str(x1_input))
    print("X2:" + str(x2_input))
    print("Таблица истинности:")
    print("OR:" + str(resultOR))
    print("AND:" + str(resultAND))


perceptron()

