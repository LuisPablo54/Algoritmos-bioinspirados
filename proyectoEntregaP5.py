'''
 Proyectos 5: ACO
 Integrantes:

    Diego Mares Rodriguez
    Luis Pablo Lopez Iracheta
    Francisco Marin Castillo

'''
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import math


def Laberinto():
    laberinto = [
    [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],  
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    ]
    return np.array(laberinto)

# Laberinto a grafo
def laberinto_a_grafo(maze):
    G = nx.Graph()
    H, W = maze.shape

    for r in range(H):
        for c in range(W):
            if maze[r, c] == 0:
                G.add_node((r, c))
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]: 
                    # Nuevas coordenadas
                    nr = r+dr
                    nc = c+dc 
                    if 0 <= nr < H and 0 <= nc < W and maze[nr, nc] == 0:
                        # Agregar arista entre nodos adyacentes para caminos
                        G.add_edge((r, c), (nr, nc))
    return G

# obtenemos los puntos de inicio y fin junto con el laberinto y el grafo
def generar_laberinto_con_camino():
    maze = Laberinto()
    G = laberinto_a_grafo(maze)
    H, W = maze.shape
    start = (0, 0)
    end = (H-1, W-1)
    return maze, G, start, end

def visualizar_ruta_final(maze, start=None, end=None, path=None, figsize=(8,8)):
    #  Visualización del laberinto y el camino
    H, W = maze.shape
    plt.figure(figsize=figsize)
    plt.imshow(maze, cmap='gray_r', interpolation='nearest')
    plt.scatter(start[1], start[0], c='green', s=120, marker='s', edgecolor='k')
    plt.scatter(end[1], end[0], c='red', s=120, marker='s', edgecolor='k')
    plt.title('Evolución del ACO', fontweight='bold', color='darkblue')
    #  camino encontrado
    py, px = zip(*path)
    plt.plot(px, py, 'b-', linewidth=2)

    plt.xticks([]); plt.yticks([])
    plt.show()


# Inicialización de feromonas en todas las aristas del grafo
def inicializar_feromonas(G, valor_inicial): 
    pheromone = {}
    for u, v in G.edges:
        pheromone[(u, v)] = valor_inicial
        pheromone[(v, u)] = valor_inicial
    return pheromone


# Para el cálculo de distancias entre puntos en el laberinto
def distancias_puntos(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) 



def construir_camino(G, origen, destino, feromona, alpha, beta, max_pasos=100):
    actual = origen
    camino = [actual]
    visitados = {actual}
    
    pasos = 0
    #  Cuando se alcanza el nodo final o se excede el número máximo de pasos
    while actual != destino and pasos < max_pasos:
        vecionos = list(G.neighbors(actual))
        disponibles = [n for n in vecionos if n not in visitados]
        
        if not disponibles:
            return None

        # Lista de probabilidades para cada nodo disponible
        probs = []
        for n in disponibles:
            try:
                tau = feromona.get((actual, n), 0.1) ** alpha
                
                # Heurística con manejo de división por cero
                distancia = distancias_puntos(n, destino)
                # Nos estaba dando error aqui con distancias cero
                if distancia == 0:
                    eta = 1.0
                else:
                    eta = (1.0 / distancia) ** beta
                
                prob_val = tau * eta
                # Asegurar que no sea negativo ni complejo
                if prob_val < 0 or math.isnan(prob_val) or math.isinf(prob_val):
                    prob_val = 0.1
                    
                probs.append(float(prob_val))
            except:
                probs.append(0.1)
        
        # Si todas las probabilidades son cero o inválidas, usar distribución uniforme
        total = sum(probs)
        if total <= 0 or math.isnan(total) or math.isinf(total):
            next_node = random.choice(disponibles)
        else:
            # Normalizar y asegurar que sean floats válidos
            probs = [max(0.0, float(p/total)) for p in probs]
            next_node = random.choices(disponibles, weights=probs)[0]
        
        camino.append(next_node)
        visitados.add(next_node)
        actual = next_node
        pasos += 1
    
    if actual == destino:
        return camino
    else:
        return None

def actualizar_feromonas(feromonas, rutas, distancias, rho, q):
    # Evaporación
    for edge in feromonas:
        feromonas[edge] *= (1 - rho)
    
    # Deposito de feromonas en todas las rutas
    for k in range(len(rutas)):
        ruta = rutas[k]
        dist = distancias[k]
    
        for i in range(len(ruta) - 1):
            a, b = ruta[i], ruta[i+1]
            
            feromonas[(a, b)] += Q / dist  # Actualización bidireccional
            feromonas[(b, a)] += Q / dist

def ejecutar_ACO(G, origen, destino, n_hormigas, n_iteraciones, alpha, beta, rho, Q):
    feromonas = inicializar_feromonas(G, valor_inicial=0.1)
    
    mejor_camino = None
    mejor_longitud = math.inf
    historial_mejores = []
    historial_promedio = []
    
    
    for iteracion in range(n_iteraciones):
        caminos = []
        longitudes = []
        
        for i in range(n_hormigas):
            camino = construir_camino(G, origen, destino, feromonas, alpha, beta)
            if camino and camino[-1] == destino:  # Solo considerar caminos que llegan al final
                longitud = len(camino) - 1
                caminos.append(camino)
                longitudes.append(longitud)
                
                if longitud < mejor_longitud and camino[-1] == destino:
                    mejor_longitud = longitud
                    mejor_camino = camino
                    
        
        # Calcular longitud promedio de los caminos exitosos
        if longitudes:
            promedio = sum(longitudes) / len(longitudes)
        else:
            promedio = math.inf
        historial_promedio.append(promedio)
        
        actualizar_feromonas(feromonas, caminos, longitudes, rho, Q)
        historial_mejores.append(mejor_longitud)
        
        # Mostrar progreso cada 10 iteraciones
        if iteracion % 10 == 0:
            caminos_exitosos = len([c for c in caminos if c is not None and c[-1] == destino])
            print(f"Iteración {iteracion}: mejor = {mejor_longitud}, promedio = {promedio:.1f}, exitosos = {caminos_exitosos}/{n_hormigas}")

    # Graficar convergencia
    plt.figure(figsize=(12, 6))
    plt.plot(historial_mejores, 'b-', linewidth=2, label='Mejor longitud')
    plt.plot(historial_promedio, 'g-', linewidth=1, alpha=0.7, label='Longitud promedio')
    plt.xlabel('Iteración')
    plt.ylabel('Longitud del camino')
    plt.title('Evolución del ACO - Exploración vs Explotación', fontweight='bold', color='darkblue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return mejor_camino, mejor_longitud

#===============Parametros del ACO=========================
maze, G, start, end = generar_laberinto_con_camino()

print("Maze shape:", maze.shape)
print("Nodos en G:", len(G.nodes))

# Parámetros ajustados para más exploración y menos convergencia rápida
numero_hormigas = 1
numero_iteraciones = 80
alpha = 1.0
beta = 3.0
rho = 0.2
Q = 0.2


best_path, best_len = ejecutar_ACO(G, start, end, numero_hormigas, numero_iteraciones, alpha, beta, rho, Q)

print(f"\nResultado final:")
print(f"Mejor camino encontrado: Longitud = {best_len}")
visualizar_ruta_final(maze, start, end, best_path) #  Visualización de la merjor ruta encontrada
