import numpy as np
import matplotlib.pyplot as plt

#########################################
#      FUNCIÓN OBJETIVO: RASTRIGIN      #
#########################################
def rastrigin(X):
    n = len(X)
    return 10 * n + np.sum(X**2 - 10 * np.cos(2 * np.pi * X))


#########################################
#        PARÁMETROS INICIALES           #
#########################################
n = 2
tam_pob = 100
generaciones = 100
Imin = -5.12
Imax = 5.12
porcentaje_seleccion = 0.4


# Parámetros de control de estancamiento
limite_estancamiento = 12   # generaciones sin mejora
no_mejora = 0               # contador
epsilon_mutacion = 0.05     # intensidad del ruido


#########################################
#     INICIALIZACIÓN DE POBLACIÓN       #
#########################################
def ini_pob(tam_pob, num_ind, Imin, Imax):
    Poblacion = np.zeros((tam_pob, num_ind))
    for i in range(tam_pob):
        for j in range(num_ind):
            Poblacion[i, j] = np.random.uniform(Imin, Imax)
    return Poblacion


#########################################
#      EVALUACIÓN DE LA POBLACIÓN       #
#########################################
def evaluar(Poblacion):
    [r, c] = Poblacion.shape
    fitness = np.zeros(r)
    for i in range(r):
        individuo = Poblacion[i, :]
        fitness[i] = rastrigin(individuo)
    return fitness


#########################################
#      SELECCIÓN DE LA ÉLITE            #
#########################################
def seleccion(Poblacion, fitness, ps):
    [r, c] = Poblacion.shape
    indices_ordenados = np.argsort(fitness)
    tamaño_elite = int(r * ps)
    Poblacion_elite = np.zeros((tamaño_elite, c))
    for i in range(tamaño_elite):
        Poblacion_elite[i, :] = Poblacion[indices_ordenados[i], :]
    return Poblacion_elite, indices_ordenados[:tamaño_elite]


#########################################
#    FUNCIÓN BUMDA MEDIA Y DESVIACIÓN   #
#########################################
def BUMDA_media_stds(Pob_elite, g, Imin, Imax, beta=1.0):
    [m, d] = Pob_elite.shape
    Z = m / ((Imax - Imin) * np.sum(np.exp(beta * g)))

    mu = np.zeros(d)
    sigma = np.zeros(d)

    for j in range(d):
        xj = Pob_elite[:, j]

        # Media
        mu_term1 = (1 / (Z * beta)) * np.sum(np.exp(beta * g) * xj)
        denom_mu = (m / ((Imax - Imin) * beta)) + np.sum(g)
        mu_term2 = np.sum(g * xj) / denom_mu
        mu[j] = mu_term1 + mu_term2

        # Varianza
        var_term1 = (1 / (Z * beta)) * np.sum(np.exp(beta * g) * (xj - mu[j])**2)
        var_term2 = np.sum(g * (xj - mu[j])**2) / np.sum(g)
        varianza = var_term1 + var_term2

        sigma[j] = np.sqrt(abs(varianza))

    return mu, sigma


#########################################
#    GENERAR NUEVA POBLACIÓN NORMAL     #
#########################################
def generar_nueva_poblacion(tam_pob, media, stds):
    dim = len(media)
    nueva_poblacion = np.zeros((tam_pob, dim))
    for i in range(tam_pob):
        for j in range(dim):
            nueva_poblacion[i, j] = np.random.normal(media[j], stds[j])
    return nueva_poblacion



############## Función de mutación leve para evitar estancamientos ##########
########esto lo que hace es que se realiza una mutacion PEQUEÑA a la poblacion con la que trabajamos para que ####
######## no esten presentes tantos estancamientos ######

def mutar_poblacion(P, intensidad=0.05, Imin = -5.12, Imax = 5.12):
    P_mutada = P.copy()
    [r, c] = P.shape
    for i in range(r):
        for j in range(c):
            # con probabilidad pequeña, se agrega ruido gaussiano
            if np.random.rand() < 0.2:
                P_mutada[i, j] += np.random.normal(0, intensidad)
                # limitar al rango permitido
                P_mutada[i, j] = np.clip(P_mutada[i, j], Imin, Imax)
    return P_mutada


#########################################
#       ALGORITMO PRINCIPAL BUMDA       #
#########################################
P = ini_pob(tam_pob, n, Imin, Imax)
mejor_solucion = None
mejor_fitness = float('inf')
historial_minimos = []


# ---------------------- Bucle principal ----------------------
for k in range(generaciones):
    fitness = evaluar(P)

    # Selección de la élite
    P_elite, idx_elite = seleccion(P, fitness, porcentaje_seleccion)

    # Cálculo de g (fitness negativo)
    g = -fitness
    g = g - np.max(g)
    g_elite = g[idx_elite]

    # Estimación de media y desviación estándar
    media, stds = BUMDA_media_stds(P_elite, g_elite, Imin, Imax)

    # Generación de nueva población
    P = generar_nueva_poblacion(tam_pob, media, stds)

    # Guardar mejor solución actual
    actual_mejor_fitness = np.min(fitness)
    actual_mejor_solucion = P[np.argmin(fitness)]
    historial_minimos.append(actual_mejor_fitness)

    # Verificar mejora
    if actual_mejor_fitness < mejor_fitness - 1e-8:
        mejor_fitness = actual_mejor_fitness
        mejor_solucion = actual_mejor_solucion
        no_mejora = 0  # se reinicia contador
    else:
        no_mejora += 1  # no hubo mejora

    # 🔁 Detección de estancamiento
    if no_mejora >= limite_estancamiento:
        print(f"⚠️ Estancamiento detectado en generación {k}. Aplicando mutación...")
        P = mutar_poblacion(P, intensidad=epsilon_mutacion, Imin=Imin, Imax=Imax)
        no_mejora = 0  # reiniciar contador tras mutación

    # Mostrar progreso cada 10 generaciones
    if k % 10 == 0 or k == generaciones - 1:
        print(f"Generación {k:03d} → Mejor fitness actual: {mejor_fitness:.6f}")



#########################################
#           GRÁFICA 3D RESULTADOS       #
#########################################
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(Imin, Imax, 200)
y = np.linspace(Imin, Imax, 200)
X, Y = np.meshgrid(x, y)
Z = 10 * 2 + (X**2 - 10 * np.cos(2 * np.pi * X)) + (Y**2 - 10 * np.cos(2 * np.pi * Y))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# Punto mínimo encontrado
ax.scatter(mejor_solucion[0], mejor_solucion[1], rastrigin(mejor_solucion),
           color='r', s=100, label='Mejor Individuo')

ax.set_title('Función de Rastrigin optimizada con BUMDA', fontsize=14)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('f(x₁, x₂)')
ax.legend()
plt.show()


# #########################################
# #       GRÁFICO DE CONVERGENCIA         #
# #########################################
# plt.figure(figsize=(8,5))
# plt.plot(historial_minimos, color='blue', linewidth=2)
# plt.title('Convergencia del algoritmo BUMDA')
# plt.xlabel('Generaciones')
# plt.ylabel('Mejor valor encontrado')
# plt.grid(True)
# plt.show()



# for k in range(generaciones):
#     fitness = evaluar(P)

#     # Selección de la élite
#     P_elite, idx_elite = seleccion(P, fitness, porcentaje_seleccion)

#     # Cálculo de g (fitness negativo)
#     g = -fitness
#     g = g - np.max(g)  # normalización para estabilidad numérica
#     g_elite = g[idx_elite]

#     # Estimación de media y desviación estándar
#     media, stds = BUMDA_media_stds(P_elite, g_elite, Imin, Imax)

#     # Generación de nueva población
#     P = generar_nueva_poblacion(tam_pob, media, stds)

#     # Guardar mejor solución actual
#     actual_mejor_fitness = np.min(fitness)
#     actual_mejor_solucion = P[np.argmin(fitness)]
#     historial_minimos.append(actual_mejor_fitness)

#     if actual_mejor_fitness < mejor_fitness:
#         mejor_fitness = actual_mejor_fitness
#         mejor_solucion = actual_mejor_solucion

#     ####Imprimir progreso cada 10 generaciones
#     if k % 10 == 0 or k == generaciones - 1:
#         print(f"Generación {k:03d} → Mejor fitness actual: {mejor_fitness:.6f}")