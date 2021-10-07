import numpy as np
import matplotlib.pyplot as plt

# Иниициализация рандомизированных входных значений
N = 100  # число экспериментов
sigma = 3  # стандартное отклонение наблюдаемых значений
k = 0.5  # теоретическое значение параметра k
b = 2  # теоретическое значение параметра b

x = np.array(range(N))  # вспомогательный вектор Х
f_x = np.array([k * z + b for z in range(N)])  # вычисляем линейную функцию f(x)=kx+b
y = f_x + np.random.normal(0, sigma, N) # случайные отклонения для моделирования


# Граффическая проверка разброса точек = красный
plt.scatter(x, y, s=2, c='red')
plt.grid(True)
#plt.show()

# вычисляем коэффициенты k и b по экспериментальным данным
m_x = x.sum() / N
m_y = y.sum() / N
a2 = np.dot(x.T, x) / N
a11 = np.dot(x.T, y) / N

kk = (a11 - m_x * m_y) / (a2 - m_x ** 2)
bb = m_y - kk * m_x


# строим точки полученной аппроксимации:
ff = np.array([kk*z+bb for z in range(N)])

# отобрааем оба линейных графика:
plt.plot(f_x, c='red')
plt.plot(ff, c='blue')
plt.show()