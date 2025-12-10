import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def controlador_pid(error, integral, error_previo, Kp, Ki, Kd, dt):

    # Término proporcional
    P = Kp * error
    
    # Término integral
    nueva_integral = integral + error * dt
    I = Ki * nueva_integral
    
    # Derivativo
    D = Kd * (error - error_previo) / dt
  
    
    señal_control = P + I + D
    señal_control = np.clip(señal_control, -10, 10)  # Limite

    
    return señal_control, nueva_integral, error

def simular_sistema(Kp, Ki, Kd, tiempo, referencia=1.0):

    # Condiciones iniciales
    y_actual = 0.0
    integral = 0.0
    error_previo = 0.0
    
    # Vectores de resultados
    salida = [y_actual]
    vector_tiempo = [tiempo[0]]
    
    dt = tiempo[1] - tiempo[0]
    
    # Simular cada paso de tiempo
    for i in range(1, len(tiempo)):
        
        error = referencia - y_actual
        
        # Calcular señal de control del PID
        u, integral, error_previo = controlador_pid(error, integral, error_previo, Kp, Ki, Kd, dt)
        
        # Ecuación dinámico 
        def dinamica_sistema(y_val, t_val, u_val):
            return -y_val + u_val
        
        # Integrar un paso de tiempo
        solucion = odeint( dinamica_sistema, y_actual, [vector_tiempo[-1], tiempo[i]], args=(u,))
        
        
        y_actual = solucion[-1][0]
        
        salida.append(y_actual)
        vector_tiempo.append(tiempo[i])
    
    return np.array(vector_tiempo), np.array(salida)

# Función objetivo
def calcular_ise(Kp, Ki, Kd, tiempo, referencia=1.0):
    t, y = simular_sistema(Kp, Ki, Kd, tiempo, referencia)
    
    # Calcular el error
    error = referencia - y
    
    # Regla del trapecio)
    ISE = np.trapz(error**2, t)
    
    # Penalizar
    if np.any(np.abs(y) > 10) or np.isnan(ISE) or np.isinf(ISE):
        return 1e6
    
    return ISE


def inicializar_pso(num_particulas, limites):
    dimensiones = len(limites)
    
    # Posiciones iniciales aleatorias
    posiciones = np.random.uniform(
        limites[:, 0], 
        limites[:, 1], 
        (num_particulas, dimensiones)
    )
    
    # Inicializar velocidades aleatorias
    velocidades = np.random.uniform(
        -1, 1, 
        (num_particulas, dimensiones)
    )
    
    return posiciones, velocidades

def actualizar_particula(posicion, velocidad, mejor_personal, mejor_global, w, c1, c2, limites):

    # Numeros aleatorios
    r1 = np.random.rand(len(posicion))
    r2 = np.random.rand(len(posicion))
    
    # Mejor de cada particula
    cognitivo = c1 * r1 * (mejor_personal - posicion)
    
    # Mejor global
    social = c2 * r2 * (mejor_global - posicion)
    
    # Actualizar velocidad
    nueva_velocidad = w * velocidad + cognitivo + social
    nueva_posicion = posicion + nueva_velocidad  # Actualizar posición
    nueva_posicion = np.clip(nueva_posicion, limites[:, 0], limites[:, 1]) #limites
    
    return nueva_posicion, nueva_velocidad

def optimizar_pso(funcion_objetivo, num_particulas=30, num_iteraciones=50, limites=None, w=0.7, c1=1.5, c2=1.5):

    if limites is None:
        limites = np.array([
            [0, 10],   
            [0, 10],   
            [0, 5]     
        ])
    
    # Inicializar
    posiciones, velocidades = inicializar_pso(num_particulas, limites)
    
    # Posiciones y puntajes
    mejores_personales = posiciones.copy()
    puntajes_personales = np.array([
        funcion_objetivo(*pos) for pos in posiciones
    ])
    
    # Mejor global
    idx_mejor_global = np.argmin(puntajes_personales)
    mejor_global = mejores_personales[idx_mejor_global].copy()
    puntaje_mejor_global = puntajes_personales[idx_mejor_global]
    
    # Historial de convergencia
    historial = [puntaje_mejor_global]
    
    # Iterar
    for iteracion in range(num_iteraciones):
        for i in range(num_particulas):
            # Actualizar partícula
            posiciones[i], velocidades[i] = actualizar_particula(
                posiciones[i], 
                velocidades[i],
                mejores_personales[i],
                mejor_global,
                w, c1, c2, 
                limites
            )
            
            # Evaluar nueva posición
            puntaje = funcion_objetivo(*posiciones[i])
            
            # Actualizar mejor personal
            if puntaje < puntajes_personales[i]:
                puntajes_personales[i] = puntaje
                mejores_personales[i] = posiciones[i].copy()
                
                # Actualizar mejor global
                if puntaje < puntaje_mejor_global:
                    puntaje_mejor_global = puntaje
                    mejor_global = posiciones[i].copy()
        
        historial.append(puntaje_mejor_global)
        
    
        if (iteracion + 1) % 5 == 0:
            print(f"Iteración {iteracion + 1}/{num_iteraciones}, " 
                  f"Mejor ISE: {puntaje_mejor_global:.6f}")
    
    return mejor_global, puntaje_mejor_global, historial

def graficar_resultados(tiempo, salida_optimizada, salida_simple, referencia, historial_pso, parametros_optimos, ise_optimo, ise_simple, config_pso):
    
    fig, ejes = plt.subplots(2, 2, figsize=(14, 10))
    
   
    ejes[0, 0].plot(tiempo, salida_optimizada, 'b-', linewidth=2, label='PID Optimizado')
    ejes[0, 0].plot(tiempo, salida_simple, 'r--', linewidth=1.5, label='Control Simple (Kp=1)')
    ejes[0, 0].axhline(y=referencia, color='g', linestyle=':', linewidth=1.5, label='Referencia')
    ejes[0, 0].set_xlabel('Tiempo (s)', fontsize=11)
    ejes[0, 0].set_ylabel('Salida y(t)', fontsize=11)
    ejes[0, 0].set_title('Respuesta del Sistema', fontsize=12, fontweight='bold')
    ejes[0, 0].legend()
    ejes[0, 0].grid(True, alpha=0.3)

    error_optimo = referencia - salida_optimizada
    error_simple = referencia - salida_simple
    ejes[0, 1].plot(tiempo, error_optimo, 'b-', linewidth=2, label='PID Optimizado')
    ejes[0, 1].plot(tiempo, error_simple, 'r--', linewidth=1.5, label='Control Simple')
    ejes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ejes[0, 1].set_xlabel('Tiempo (s)', fontsize=11)
    ejes[0, 1].set_ylabel('Error e(t)', fontsize=11)
    ejes[0, 1].set_title('Error vs Tiempo', fontsize=12, fontweight='bold')
    ejes[0, 1].legend()
    ejes[0, 1].grid(True, alpha=0.3)

    ejes[1, 0].plot(historial_pso, 'b-', linewidth=2)
    ejes[1, 0].set_xlabel('Iteración', fontsize=11)
    ejes[1, 0].set_ylabel('Mejor ISE', fontsize=11)
    ejes[1, 0].set_title('Convergencia del PSO', fontsize=12, fontweight='bold')
    ejes[1, 0].grid(True, alpha=0.3)
    ejes[1, 0].set_yscale('log')
    
    ejes[1, 1].axis('off')
    Kp_opt, Ki_opt, Kd_opt = parametros_optimos
    texto_info = f"""
Parametros:

Kp (Proporcional): {Kp_opt:.4f}
Ki (Integral): {Ki_opt:.4f}
Kd (Derivativo): {Kd_opt:.4f}

ISE (PID Opt): {ise_optimo:.6f}
ISE (Simple): {ise_simple:.6f}

"""
    ejes[1, 1].text(0.1, 0.5, texto_info, fontsize=15, family='monospace',
                    verticalalignment='center', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout() 
    plt.show()


# Configuración del problema
tiempo_final = 20.0
tiempo = np.linspace(0, tiempo_final, 1000)
referencia = 1.0

print(f"Referencia: escalón unitario (y_ref = {referencia})")
print(f"Objetivo: Minimizar ISE (Error Cuadrático Integral)")
print("=" * 60)

# Definir función objetivo para PSO
def objetivo(Kp, Ki, Kd):
    return calcular_ise(Kp, Ki, Kd, tiempo, referencia)

# Configuración del PSO
config_pso = {
    'num_particulas': 10,
    'num_iteraciones': 30,
    'w': 0.7,
    'c1': 1.5,
    'c2': 1.5
}

# Ejecutar optimización PSO
parametros_optimos, ise_minimo, historial = optimizar_pso(
    objetivo,
    num_particulas=config_pso['num_particulas'],
    num_iteraciones=config_pso['num_iteraciones'],
    w=config_pso['w'],
    c1=config_pso['c1'],
    c2=config_pso['c2']
)

Kp_optimo, Ki_optimo, Kd_optimo = parametros_optimos

# Mostrar resultados
print("\n" + "=" * 60)
print("Resultados de PSO")
print("=" * 60)
print(f"Kp óptimo: {Kp_optimo:.4f}")
print(f"Ki óptimo: {Ki_optimo:.4f}")
print(f"Kd óptimo: {Kd_optimo:.4f}")
print(f"ISE mínimo: {ise_minimo:.6f}")
print("=" * 60)


t_optimo, y_optima = simular_sistema(
    Kp_optimo, Ki_optimo, Kd_optimo, tiempo, referencia
)


t_simple, y_simple = simular_sistema(1, 0, 0, tiempo, referencia)
ise_simple = calcular_ise(1, 0, 0, tiempo, referencia)

# Generar gráficas
graficar_resultados(
    t_optimo, y_optima, y_simple, referencia, historial,
    parametros_optimos, ise_minimo, ise_simple, config_pso
)

print("\nFin")



