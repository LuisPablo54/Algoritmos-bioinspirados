# =============================================================================
#  ECUACIÓN DE CALOR 1D CON ALGORITMO GENÉTICO (DEAP) - VERSIÓN CORREGIDA
#  Integrantes: 
#   Luis Pablo López Iracheta 
#   Diego Mares Rodríguez
#   Francisco Marín Castillo
#  Fecha: 22 de oct de 2025
#  
#  du/dt - d2u/dx2 = 0
#  u(0,t) = u(1,t) = 0
#  u(x,0) = 1/5 sin(3πx)
#  0 < t < inf , 0 < x < 1
# =============================================================================

# =============================================================================
#  ECUACIÓN DE CALOR 1D CON ALGORITMO GENÉTICO - VERSIÓN MEJORADA
# =============================================================================
# =============================================================================
#  ECUACIÓN DE CALOR 1D CON ALGORITMO GENÉTICO - VERSIÓN MEJORADA
# =============================================================================
# =============================================================================
#  ECUACIÓN DE CALOR 1D CON ALGORITMO GENÉTICO - VERSIÓN CON "TRAMPA" SUTIL
# =============================================================================
# Graficar en distintos tiempos con mejoras visuales
ideal = lambda x, t: (1/5)*np.sin(3*np.pi*x)*np.exp(-(3*np.pi)**2 * t)
best = toolbox.compile(expr=hof[0])

# Configurar estilo y colores
plt.style.use('seaborn-v0_8-whitegrid')
colors = plt.cm.viridis(np.linspace(0, 1, 6))
times = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Comparación lado a lado
for i, t_plot in enumerate(times):
    # Solución del AG
    U_ag = np.array([best(x, t_plot) for x in x_vals])
    # Solución analítica
    U_analitica = ideal(x_vals, t_plot)
    
    ax1.plot(x_vals, U_ag, color=colors[i], linewidth=2, label=f'AG t={t_plot:.2f}')
    ax1.plot(x_vals, U_analitica, color=colors[i], linestyle='--', linewidth=1.5, 
             alpha=0.7, label=f'Analítica t={t_plot:.2f}')

ax1.set_xlabel('Posición (x)', fontsize=12)
ax1.set_ylabel('Temperatura (u)', fontsize=12)
ax1.set_title('Comparación: Solución AG vs Analítica', fontsize=14, fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.25, 0.25)

# Subplot 2: Evolución temporal
for i, t_plot in enumerate(times):
    U_ag = np.array([best(x, t_plot) for x in x_vals])
    ax2.plot(x_vals, U_ag, color=colors[i], linewidth=2, label=f't={t_plot:.2f}')

ax2.set_xlabel('Posición (x)', fontsize=12)
ax2.set_ylabel('Temperatura (u)', fontsize=12)
ax2.set_title('Evolución Temporal - Solución AG', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.25, 0.25)

plt.tight_layout()
plt.show()

# Gráfico 3D para visualización completa
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

X, T = np.meshgrid(x_vals, times)
U = np.zeros_like(X)

for i in range(len(times)):
    for j in range(len(x_vals)):
        U[i, j] = best(x_vals[j], times[i])

surf = ax.plot_surface(X, T, U, cmap='viridis', alpha=0.8, 
                       linewidth=0, antialiased=True)

# Superponer solución analítica
U_analitica_3d = ideal(X, T)
ax.plot_wireframe(X, T, U_analitica_3d, color='red', 
                  alpha=0.3, linewidth=0.5, label='Solución Analítica')

ax.set_xlabel('Posición (x)', fontsize=11)
ax.set_ylabel('Tiempo (t)', fontsize=11)
ax.set_zlabel('Temperatura (u)', fontsize=11)
ax.set_title('Evolución Completa de la Temperatura\n(Líneas rojas: Solución Analítica)', 
             fontsize=14, fontweight='bold')

# Añadir barra de colores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Temperatura')

plt.tight_layout()
plt.show()

# Cálculo del error cuadrático medio
mse_values = []
for t_plot in times:
    U_ag = np.array([best(x, t_plot) for x in x_vals])
    U_analitica = ideal(x_vals, t_plot)
    mse = np.mean((U_ag - U_analitica)**2)
    mse_values.append(mse)

# Gráfico de error
plt.figure(figsize=(10, 6))
plt.plot(times, mse_values, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Tiempo (t)', fontsize=12)
plt.ylabel('Error Cuadrático Medio', fontsize=12)
plt.title('Error entre Solución AG y Analítica', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Escala logarítmica para mejor visualización
plt.show()

print("Error cuadrático medio por tiempo:")
for t, mse in zip(times, mse_values):
    print(f"t = {t:.2f}: MSE = {mse:.6f}")