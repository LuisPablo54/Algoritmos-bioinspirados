"""
Proyecto de Entrega P3
Integrantes:
- Luis Pablo López Iracheta
- Diego Mares Rodríguez
- Francisco Marín Castillo
"""

########################  Ecuación de Calor 1D con Algoritmo Genético (DEAP)
######################  Mejora visual: curvas, convergencia

import math, random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms   # Librería DEAP para GA

########### 1# Parámetros del problema #################
##########################

K = 5                 # Número de modos seno (términos de la serie)
Tmax = 0.5            # Tiempo máximo a simular
Nx, Nt = 64, 32       # Número de puntos espaciales y temporales
L_PDE, L_IC = 1.0, 1.0  # Pesos para los términos de la función de pérdida

rng = random.Random(42)  # Semilla aleatoria reproducible
np.random.seed(42)

############## 2# Rejilla de evaluación ####################
x = np.linspace(0.0, 1.0, Nx)               # Discretización espacial
t = np.linspace(0.0, Tmax, Nt)              # Discretización temporal
X, T = np.meshgrid(x, t, indexing="ij")     # Malla 2D de evaluación (x,t)

IC_target = (1.0/5.0) * np.sin(3.0*np.pi*X[:, 0])  # Condición inicial u(x,0)


################## 3# Modelo paramétrico #####################
def u_xt(params, X, T):
    """Función candidata u(x,t) con parámetros libres (a_k, b_k)."""
    a = np.array(params[:K])                 # Coeficientes de amplitud
    b_raw = np.array(params[K:2*K])          # Coeficientes de decaimiento
    b = np.abs(b_raw)                        # Se fuerza b >= 0 (estabilidad)
    U = np.zeros_like(X)
    for k in range(1, K+1):
        # Serie seno con decaimiento temporal exponencial
        U += a[k-1] * np.exp(-b[k-1]*T) * np.sin(k*np.pi*X)
    return U

def residual_pde(params, X, T):
    """Calcula el residual de la ecuación de calor: R = u_t - u_xx."""
    a = np.array(params[:K])
    b = np.abs(np.array(params[K:2*K]))
    R = np.zeros_like(X)
    for k in range(1, K+1):
        s = np.sin(k*np.pi*X)
        e = np.exp(-b[k-1]*T)
        ut  = (-b[k-1] * a[k-1]) * e * s             # Derivada temporal u_t
        uxx = (-(k*np.pi)**2 * a[k-1]) * e * s       # Segunda derivada espacial u_xx
        R += ut - uxx
    return R

def loss_total(params):
    """Función objetivo (fitness): combina error PDE + condición inicial."""
    U = u_xt(params, X, T)                 # Solución generada
    R = residual_pde(params, X, T)         # Residual de la PDE
    pde_mse = np.mean(R**2)                # Error promedio del residual
    U0 = U[:, 0]                           # Perfil inicial u(x,0)
    ic_mse = np.mean((U0 - IC_target)**2)  # Error con respecto a la IC
    return L_PDE * pde_mse + L_IC * ic_mse # Pérdida total



# ######################## 4# DEAP: definición del GA ##############################
# Se define el tipo de individuo y la estrategia de minimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))     # Fitness a minimizar
creator.create("Individual", list, fitness=creator.FitnessMin)  # Individuo tipo lista
toolbox = base.Toolbox()

B_MAX = ((K+3)**2) * (math.pi**2)   # Cota superior de b_k para exploración
def init_gene_a(): return rng.uniform(-1.0, 1.0)   # a_k aleatorios [-1,1]
def init_gene_b(): return rng.uniform(0.0, B_MAX)  # b_k aleatorios [0, B_MAX]



def init_individual():
    """Crea un individuo con K valores a_k y K valores b_k."""
    genes = [init_gene_a() for _ in range(K)] + [init_gene_b() for _ in range(K)]
    return creator.Individual(genes)

# Registro de operadores en DEAP
toolbox.register("individual", init_individual)               # Crea un individuo
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # Crea población
toolbox.register("evaluate", lambda ind: (loss_total(ind),))  # Evalúa el fitness
toolbox.register("mate", tools.cxBlend, alpha=0.5)            # Cruce de genes
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.25, indpb=0.2)  # Mutación gaussiana
toolbox.register("select", tools.selTournament, tournsize=3)  # Selección por torneo

# ################### 5# Entrenamiento #######################
def run_ga(n_pop=120, n_gen=200, cxpb=0.6, mutpb=0.3, seed=42):
    """Ejecución del algoritmo genético con los parámetros dados."""
    rng.seed(seed)
    pop = toolbox.population(n=n_pop)   # Genera población inicial

    # Estadísticas (mínimo y promedio del fitness)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    hof = tools.HallOfFame(1)  # Guarda el mejor individuo encontrado

    # Bucle evolutivo principal (usa el algoritmo simple de DEAP)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                   ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
    return hof[0], log   # Retorna el mejor individuo y el registro de evolución



############################## 6# Ejecución + la visualización ####################
#######################################


if __name__ == "__main__":
    best, log = run_ga()  # Entrena el algoritmo y obtiene el mejor resultado

    # Parámetros óptimos encontrados
    a_hat = np.array(best[:K])
    b_hat = np.abs(np.array(best[K:2*K]))
    print("\n==== Mejor individuo ====")
    for k in range(1, K+1):
        print(f"k={k:>2d}: a_k = {a_hat[k-1]: .6f}   b_k = {b_hat[k-1]: .6f}   "
              f"( (k*pi)^2 = {(k*math.pi)**2: .6f} )")
    print(f"Pérdida total: {loss_total(best):.6e}")

    ######## Comparación con la solución analítica ########
    
    U_est = u_xt(best, X, T)  # Solución encontrada por GA
    # Solución exacta analítica para comparación
    U_true = (1.0/5.0) * np.exp(-((3.0*math.pi)**2) * T) * np.sin(3.0*math.pi * X)

    ########### Errores globales################
    
    mse_true = np.mean((U_est - U_true)**2)
    rel_err = np.linalg.norm(U_est - U_true) / np.linalg.norm(U_true)
    print(f"MSE vs. solución analítica: {mse_true:.6e}")
    print(f"Error relativo L2: {rel_err:.6e}")

    ################ Gráfica 01: Convergencia del fitness ###################
    gens = log.select("gen")          # Generaciones
    min_fit = log.select("min")       # Mejor fitness por generación
    avg_fit = log.select("avg")       # Fitness promedio

    plt.figure(figsize=(6,4))
    plt.plot(gens, min_fit, label="Fitness mínimo")
    plt.plot(gens, avg_fit, label="Fitness promedio")
    plt.xlabel("Generación"); plt.ylabel("Error")
    plt.title("Convergencia del algoritmo genético")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

    #################### Gráfica 02: Comparación 1D (tiempo medio) ####################
    j = Nt // 2   # Toma el tiempo medio
    plt.figure(figsize=(6,4))
    plt.plot(x, U_true[:, j], 'k', label="Analítica (t≈medio)")
    plt.plot(x, U_est[:, j], '--r', label="GA-DEAP (t≈medio)")
    plt.xlabel("x"); plt.ylabel("u(x,t)")
    plt.title("Comparación analítica vs. GA (tiempo medio)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

    ############## Gráfica 03: Evolución temporal en varios instantes ##################
    fig, ax3 = plt.subplots(figsize=(6,4))
    tiempos_deseados = [0.0, 0.001, 0.02, 0.03, 0.05]  # Instantes a visualizar
    time_indices = [np.argmin(np.abs(t - td)) for td in tiempos_deseados]  # Índices más cercanos
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    line_styles = ['-', '--', '-.', ':', (0, (3,1,1,1))]

    for i, idx in enumerate(time_indices):
        t_val = t[idx]
        ax3.plot(x, U_est[:, idx],
                 color=colors[i % len(colors)],
                 linestyle=line_styles[i % len(line_styles)],
                 linewidth=2,
                 label=f't = {t_val:.3f} s')

    ax3.set_xlabel("Posición x")
    ax3.set_ylabel("Temperatura u(x,t)")
    ax3.set_title("Evolución temporal de la solución GA-DEAP")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    ############ Resumen final ###########
    ########## Error relativo y la perdida total final##############
    
    print("\nResumen:")
    print(f"• MSE total: {mse_true:.3e}")
    print(f"• Error relativo L2: {rel_err:.3e}")
    print(f"• Pérdida total final: {loss_total(best):.3e}")
  
