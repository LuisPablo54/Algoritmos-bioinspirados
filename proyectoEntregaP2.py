import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#  Función Rastrigin
def rastrigin(x):
    return 10 * len(x) + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x])

# Inicialización de la población
def ini_poblacion(tam_poblacion, num_ind, Imin, Imax):
    poblacion = np.zeros((tam_poblacion, num_ind))
    for i in range(tam_poblacion):
        for j in range(num_ind):
            poblacion[i][j] = np.random.uniform(Imin, Imax)
    return poblacion

# Evaluación de la población
def fitness(poblacion, funcion=rastrigin):
    fitness_values = np.zeros(len(poblacion))
    for i in range(len(poblacion)):
        fitness_values[i] = funcion(poblacion[i])
    return fitness_values

# Selección de la población elite 
def seleccion(Poblacion, fitness, ps):
    num_elite = int(len(Poblacion) * ps) 
    idx = np.argsort(fitness) # Ordena los índices por fitness de menor es mejor
    poblacionElite = Poblacion[idx[:num_elite]] 
    fitnessElite = fitness[idx[:num_elite]]
    
    return poblacionElite, fitnessElite # Devuelve también los fitness de la elite

def BUMDA_media_stds(pobla_elite, g, Imin, Imax, sigma_floor=0.01, beta=1.0, eps=1e-8):
    # pobla_elite: matriz de individuos elite
    # g: vector de fitness correspondiente
    # Imin, Imax: límites del espacio de búsqueda
    # Sigma_floor: desviación mínima absoluta
    # beta: parámetro de selección por peso
    # eps: para evitar división por cero
    m, d = pobla_elite.shape
    w = np.exp(-beta * np.array(g, dtype=float)) # pesos basados en fitness
    w_sum = np.sum(w) # suma de pesos
    # Normaliza pesos
    # evita división por cero
    if w_sum == 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w_sum

    mu = np.zeros(d) # media ponderada
    sigma = np.zeros(d) # desviación estándar

    # Calcula media y desviación para cada dimensión
    for j in range(d):
        xj = pobla_elite[:, j]
        mu_j = np.sum(w * xj)
        var_j = np.sum(w * (xj - mu_j) ** 2)
        sigma_j = np.sqrt(max(var_j, eps))

        # límite superior y ahora límite inferior
        max_sigma = (Imax - Imin) / 2.0
        sigma_j = min(sigma_j, max_sigma)

        # floor absoluto calculado desde fracción del rango
        sigma_j = max(sigma_j, sigma_floor)

        mu[j] = mu_j
        sigma[j] = sigma_j

    return mu, sigma


def generar_nueva_poblacion(tam_pob, media, stds): 
    dim = len(media)
    nueva_poblacion = np.zeros((tam_pob, dim))
    for i in range(tam_pob):
        for j in range(dim):
            nueva_poblacion[i, j] = np.random.normal(media[j], stds[j])
    return nueva_poblacion

#----------- Parámetros y ejecución del algoritmo -----------
tam_poblacion = 100 
num_ind = 2
generaciones = 100
Imin = -5.12
Imax = 5.12
ps = 0.5

sigma_floor_rel = 0.01   # fracción del rango total como sigma mínima (1%)
sigma_floor = (Imax - Imin) * sigma_floor_rel

beta = 0.5   # Beta que determina la presión de selección y que tan rápido converge
ps = 0.5
P = ini_poblacion(tam_poblacion, num_ind, Imin, Imax)
mejor_solucion = None
mejor_fitness = float('inf')

for gen in range(generaciones):
    # Evalúa la población actual
    fitness_values = fitness(P)

    # Actualiza mejor con respecto a la población evaluada (no la generada después)
    idx_best = np.argmin(fitness_values)
    if fitness_values[idx_best] < mejor_fitness:
        mejor_fitness = fitness_values[idx_best]
        mejor_solucion = P[idx_best].copy()

    # Selecciona la población elite
    poblacionElite, fitnessElite = seleccion(P, fitness_values, ps)

    # Calcula media y desviación con sigma_floor en unidades absolutas
    mu, sigma = BUMDA_media_stds(poblacionElite, fitnessElite, Imin, Imax,
                                beta=beta, sigma_floor=sigma_floor)

    # Genera nueva poblacion muestreando normales y aplicando clip
    P_new = generar_nueva_poblacion(tam_poblacion, mu, sigma)
    P_new = np.clip(P_new, Imin, Imax)

    # Elitismo: conserva el mejor individuo actual en la nueva población
    P_new[0] = mejor_solucion.copy()

    P = P_new

    if gen % 20 == 0:
        print(f"Generación {gen}: Mejor fitness = {mejor_fitness}")
print(f"Mejor solución encontrada: {mejor_solucion} con el fitness = {mejor_fitness}")
print("Espacio en 3D", (f"{mejor_solucion[0]:.4f}", f"{mejor_solucion[1]:.4f}", f"{rastrigin(mejor_solucion):.4f}"))

# Graficas
x = np.linspace(Imin, Imax, 400)
y = np.linspace(Imin, Imax, 400)
X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])
fig = plt.figure(figsize=(14, 6))
cmap_3d = 'cool'       
cmap_2d = 'PuBu'      
# ---------------------- GRÁFICA 3D  ----------------
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Función Rastrigin 3D', fontsize=14, fontweight='bold')
surface = ax1.plot_surface(X, Y, Z, cmap=cmap_3d, alpha=0.85, linewidth=0, antialiased=True, rstride=2, cstride=2)
# Punto de la mejor solución
ax1.scatter(mejor_solucion[0], mejor_solucion[1], rastrigin(mejor_solucion), color='red', s=150, label='Mejor solución', edgecolor='black', linewidth=2)

# Configuración de ejes
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_zlabel('f(x, y)', fontsize=12)
ax1.legend()
cbar = fig.colorbar(surface, ax=ax1, shrink=0.5, aspect=20, pad=0.1)
cbar.set_label('Valor de f(x, y)', fontsize=10)

# ---------------------- GRÁFICA 2D  ----------------
ax2 = fig.add_subplot(122)
ax2.set_title('Función Rastrigin 2D', fontsize=14, fontweight='bold')

# Contorno
contour = ax2.contour(X, Y, Z, 25, cmap=cmap_2d, linewidths=1.5)
contourf = ax2.contourf(X, Y, Z, 25, cmap=cmap_2d, alpha=0.7)
ax2.plot(mejor_solucion[0], mejor_solucion[1], 'ro', markersize=12, label='Mejor solución', markeredgecolor='black', markeredgewidth=2)
ax2.plot(0, 0, 'go', markersize=12, label='Mínimo global', markeredgecolor='black', markeredgewidth=2)

# Configuración de ejes
ax2.set_xlabel('X1', fontsize=12)
ax2.set_ylabel('X2', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Barra de colores para el contorno
cbar2 = fig.colorbar(contourf, ax=ax2, shrink=0.8)
cbar2.set_label('Valor de f(X1, X2)', fontsize=10)

plt.tight_layout()
plt.show()