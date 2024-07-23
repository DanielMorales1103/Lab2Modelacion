import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Datos iniciales
y0 = 15
y4 = 56
N = 300
t0 = 0
t4 = 4
t12 = 12

# Definimos la función para encontrar k
def find_k(k):
    return y0 * np.exp(k * t4) / (1 + (y0 / N) * (np.exp(k * t4) - 1)) - y4

# Encontramos k usando fsolve
k_initial_guess = 0.1
k = fsolve(find_k, k_initial_guess)[0]

# Método de Euler
def euler_method(k, y0, N, t0, t_final, step_size):
    t_values = np.arange(t0, t_final + step_size, step_size)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + step_size * k * (1 - y_values[i-1] / N) * y_values[i-1]
    
    return t_values, y_values

# Aplicamos el método de Euler
t_values, y_values = euler_method(k, y0, N, t0, t12, 1)

# Imprimimos el resultado
y_12_days = y_values[-1]
print(f'La población después de 12 días es aproximadamente {y_12_days:.2f} mariposas.')

# Encontramos el tiempo en que la población es de 150 mariposas
def find_time_to_150(y_values, t_values):
    for t, y in zip(t_values, y_values):
        if y >= 150:
            return t
    return None

time_to_150 = find_time_to_150(y_values, t_values)
print(f'La población alcanza 150 mariposas en aproximadamente {time_to_150} días.')

# Graficamos los resultados
plt.plot(t_values, y_values, label='Población de mariposas')
plt.axhline(150, color='r', linestyle='--', label='Crecimiento máximo de población (150 mariposas)')
plt.xlabel('Días')
plt.ylabel('Población')
plt.legend()
plt.title('Crecimiento poblacional de mariposas')
plt.show()
