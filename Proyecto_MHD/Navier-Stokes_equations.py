'''

Navier-Stokes equations:
Las ecuaciones de Navier-Stokes son un conjunto de ecuaciones diferenciales 
parciales no lineales que describen el movimiento de un fluido viscoso

Estas ecuaciones expresan la conservación del momento y la conservación de
la masa para un fluido en movimiento, y se utilizan para modelar una 
amplia variedad de fenómenos, desde el flujo de aire alrededor de un 
avión hasta el movimiento del agua en un río

Código de prueba


'''
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Constantes

N_POINTS = 41
DOMAIN_SIZE = 1.0
N_ITERATIONS = 500
TIME_STEP_LENGTH = 0.001
KINEMATIC_VISCOSITY = 0.1
DENSITY = 1.0
HORINZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATION = 50

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1) # Longitud del elemento
    # Crear la malla
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    X, Y = np.meshgrid(x,y) # Coordenadas de la malla

    u_preb = np.zeros_like(X) # Velocidad en x
    v_preb = np.zeros_like(X) # Velocidad en y
    p_preb = np.zeros_like(X) # Presion

    # Central difference function es para la derivada en x
    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 2:] 
            - 
            f[1:-1, 0:-2]
        ) /  (
            2 * element_length
        )
        return diff # Devuelve la diferencia central en x
    
    # Central difference function es para la derivada en y
    def central_difference_y(f):
        diff = np.zeros_like(f) 
        diff[1:-1, 1:-1] = ( 
            f[2:, 1:-1] 
            - 
            f[0:-2, 1:-1]
        ) /  (
            2 * element_length
        )
        
        return diff # Devuelve la diferencia central en y
    
    # Laplace function es para la segunda derivada
    def laplace(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            f[1:-1, 0:-2] 
            + 
            f[0:-2, 1:-1]
            - 
            4
            * 
            f[1:-1, 1:-1]
            +
            f[1:-1, 2:]
            +
            f[2:, 1:-1]
        ) / (
            element_length ** 2
            )

        return diff
    
    maximum_posible_time_step_length = (
        0.5 * element_length ** 2 / KINEMATIC_VISCOSITY
    )
    if TIME_STEP_LENGTH > maximum_posible_time_step_length:
        raise ValueError(
            f"El tiempo de paso es muy grande para la estabilidad numérica. "
            f"Debe ser menor a {maximum_posible_time_step_length}"
        )

    for i in tqdm(range(N_ITERATIONS)):
        # Se calculan las derivadas primeras
        # 'u' y 'v' son las velocidades en x y y respectivamente
        d_u_prev_d_x = central_difference_x(u_preb) # Derivada en x de u
        d_u_prev_d_y = central_difference_y(u_preb) # Derivada en y de u
        d_v_prev_d_x = central_difference_x(v_preb) # Derivada en x de v
        d_v_prev_d_y = central_difference_y(v_preb) # Derivada en y de v

        # Se calcula el laplace de u y v, 
        # lo que es la segunda derivada
        laplace_u_prev = laplace(u_preb) # Laplace de u
        laplace_v_prev = laplace(v_preb) # Laplace de v

        # ***********Se calcula la presion***************
        # Se calcula la velocidad en x
        u_tent = (
            u_preb
            +
            TIME_STEP_LENGTH * (
                -
                (
                    u_preb * d_u_prev_d_x
                    +
                    v_preb * d_u_prev_d_y
                )
            +
            KINEMATIC_VISCOSITY  * laplace_u_prev
            )
            )
        # Se calcula la velocidad en y 
        v_tent = ( 
            v_preb
            +
            TIME_STEP_LENGTH * (
                -
                (
                    u_preb * d_v_prev_d_x
                    +
                    v_preb * d_v_prev_d_y
                )
            +
            KINEMATIC_VISCOSITY  * laplace_v_prev
            )
        )
        #***************************************
       
        # Velocidades boundary conditions: Homogeneas en todos lados
        # excepto en la parte superior, la cual tiene una velocidad fija
       
        u_tent[0, :] = 0.0 # Parte inferior
        u_tent[:, 0] = 0.0 # Lado izquierdo
        u_tent[:, -1] = 0.0 # Lado derecho
        u_tent[-1, :] = HORINZONTAL_VELOCITY_TOP # Parte superior
        # Con v_tent igual a cero en todos lados
        v_tent[0, :] = 0.0 # Parte inferior
        v_tent[:, 0] = 0.0 # Lado izquierdo
        v_tent[:, -1] = 0.0 # Lado derecho
        v_tent[-1, :] = 0.0 # Parte superior

        # Se calculan las derivadas primeras de u_tent y v_tent
        d_u_tent_d_x = central_difference_x(u_tent)
        d_v_tent_d_y = central_difference_y(v_tent)

        # Computar la presion correccion usando la ecuacion de Poisson
        rhs = ( 
            DENSITY / TIME_STEP_LENGTH * (
                d_u_tent_d_x
                +
                d_v_tent_d_y
            )
        )

        # Iterar para resolver la ecuacion de Poisson para la presion
        for _ in range(N_PRESSURE_POISSON_ITERATION):
            p_next = np.zeros_like(p_preb)
            p_next[1:-1, 1:-1] = 1/4 *  (
                
                    p_preb[1:-1, 0:-2]
                    +
                    p_preb[0:-2, 1:-1]
                    +
                    p_preb[1:-1, 2:]
                    +
                    p_preb[2:, 1:-1]
                    - 
                    element_length ** 2 
                    *
                    rhs[1:-1, 1:-1]
            )
                
            # Boundary conditions para la presion: Neumann en todos lados
            # excepto en la parte superior, donde es Dirichlet homogenea
            p_next[:, -1] = p_next[:, -2] # Lado derecho
            p_next[0, :] = p_next[1, :] # Parte inferior
            p_next[:, 0] = p_next[:, 1] # Lado izquierdo
            p_next[-1, :] = 0.0 # Parte superior


        # Actualizar p_preb
        p_preb = p_next

        d_p_next_d_x = central_difference_x(p_next) # Derivada en x de p
        d_p_next_d_y = central_difference_y(p_next) # Derivada en y de p

            # Correccion de las velocidades debido a que el fluido es incompresible
        u_next = (
                u_tent
                -
                TIME_STEP_LENGTH / DENSITY
                *
                d_p_next_d_x
            )
        v_next = (
                v_tent
                -
                TIME_STEP_LENGTH / DENSITY
                *
                d_p_next_d_y
            )
        
        u_next[0, :] = 0.0 # Parte inferior
        u_next[:, 0] = 0.0 # Lado izquierdo
        u_next[:, -1] = 0.0 # Lado derecho
        u_next[-1, :] = HORINZONTAL_VELOCITY_TOP # Parte superior
        # Con v_tent igual a cero en todos lados
        v_next[0, :] = 0.0 # Parte inferior
        v_next[:, 0] = 0.0 # Lado izquierdo
        v_next[:, -1] = 0.0 # Lado derecho
        v_next[-1, :] = 0.0 # Parte superior

        # Actualizamos el tiempo
        u_preb = u_next
        v_preb = v_next
        p_preb = p_next
    # Graficar los resultados
    plt.figure(figsize=(11,7), dpi=100)
    plt.contourf(X, Y, p_preb, alpha=0.5, cmap=plt.cm.viridis)
    plt.colorbar()

    #plt.quiver(X, Y, u_next, v_next, color = 'black')
    plt.streamplot(X, Y, u_next, v_next, color='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Campo de Velocidades')
    plt.show()


if __name__ == "__main__":
    main()
