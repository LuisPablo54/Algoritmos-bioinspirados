

import numpy as np
import pandas as pd


#*****************************Class of algoritm stand binary for rank***********************************
class cl_alg_stn_bin_rank():
    def __init__(self, funtion, population, cant_genes, cant_ciclos, 
                 selection_percent, crossing, mutation_percent, i_min,
                    i_max, optimum):
        self.funtion = funtion
        self.population = population
        self.cant_genes = cant_genes
        self.cant_ciclos = cant_ciclos
        self.selection_percent = selection_percent
        self.crossing = crossing
        self.mutation_percent = mutation_percent
        self.i_min = i_min
        self.i_max = i_max
        self.optimum = optimum

    def run(self):
        print(f"\n[INFO] Starting algorithm: Standard Binary for Rank")
        self.bin_population = self.create_binary_population(self.population, self.cant_genes)
        for i in range(self.cant_ciclos):
            self.Crossing_bin = self.crossing_binary_population(self.bin_population, self.crossing)
            self.Mutations_bin = self.mutations_binary_population(self.Crossing_bin, self.mutation_percent)

            self.select_bin = self.seleccionRanking(self.Mutations_bin, self.selection_percent, self.i_min, self.i_max, self.optimum)

            self.bin_population = self.select_bin.copy()

        best_population_decimal = self.decode_binary_population(self.bin_population, self.i_min, self.i_max)
        best_individual = best_population_decimal[0]
        print(f"[INFO] Best solution found: {best_individual} with fitness: {self.fitness_binary_population(self.bin_population, self.i_min, self.i_max).max()}")
        print(f"[INFO] Ending algoritm stand binary for rank\n")
        return best_individual


    def create_binary_population(self, n, l):
        # n: número de individuos
        # l: longitud de la representación binaria
        self.bin_population = np.random.randint(0, 2, size=(n, l))
        return self.bin_population

    def decode_binary_population(self, P, Imin, Imax):
        # P: population binary
        # Imin: min interval
        # Imax: max interval
        [r, c] = P.shape # r: number of individuals, c: number of genes
        decimal = np.zeros(r)
        rescaled_decimal = np.zeros(r)

        for i in range(r):
            for j in range(c):
                # Transform from binary to integer decimal
                decimal[i] = decimal[i] + P[i, j] * 2 ** (c - j - 1)
                # Rescale the decimal value in the search space (0 to 2)
                rescaled_decimal[i] = (Imax - Imin) * decimal[i] / (2 ** c - 1) + Imin
        return rescaled_decimal

    def fitness_binary_population(self, population, Imin, Imax):
        # [r, c] = population.shape
        x = self.decode_binary_population(population, Imin, Imax)
        fitness = self.funtion(x)
        return fitness
    

    def crossing_binary_population(self, population, cross_percent):
        [r, c] = population.shape
        # cross_percent: percent of population to cross, between 0 and 0.5
        cross_percent = float(cross_percent)
        num_cross = int(r * cross_percent)
        AuxMatrix = np.zeros((2 * num_cross, c))

        for i in range(num_cross):
            r1 = np.random.randint(0, r, size=(1, 2)) # Select two random parents

            # Select the parents to cross
            father1 = population[r1[0, 0], :]
            father2 = population[r1[0, 1], :]

            # Select the crossing point
            r2 = np.random.randint(0, c)

            # Create the children
            children1 = np.concatenate((father1[0:r2], father2[r2:]))
            children2 = np.concatenate((father2[0:r2], father1[r2:]))

            # Save the children in the auxiliary matrix
            AuxMatrix[2 * i, :] = children1
            AuxMatrix[2 * i + 1, :] = children2
        return AuxMatrix


    def mutations_binary_population(self, population, mutation_percent):
        # population: binary population
        # mutation_percent: percent of genes to mutate, between 0 and 0.5
        [r, c] = population.shape
        n = int(mutation_percent * c * r) # Number of genes to mutate
        for i in range(n):
            r1 = np.random.randint(0, r) # Number aleatory to select the individual to mutate
            r2 = np.random.randint(0, c) # Number aleatory to select the gene to mutate
            
            # Compare the value of the gene and change it
            if (population[r1, r2] == 0):
                population[r1, r2] = 1
            else:
                population[r1, r2] = 0

        return population
    
    def seleccionRanking(self, poblacion, select_percent, Imin, Imax, optimum='max'):
        [r, c] = poblacion.shape
        pnew = np.zeros((r, c))  # Matrix to save the new population
        n = int(select_percent * r)  # Number of individuals to select

        fitness = self.fitness_binary_population(poblacion, Imin, Imax).reshape(r, 1)  # Se agrega una columna para guardar su valor Fitness
        expanded_population = np.concatenate([poblacion, fitness], axis=1)  # axis = 0 -> son los renglones, 1 son las columnas

        if (optimum == 'max'):
            indices = np.argsort(expanded_population[:, -1])[::-1]
        elif (optimum == 'min'):
            indices = np.argsort(expanded_population[:, -1])
        else:
            raise ValueError("The optimum parameter must be 'max' or 'min'")
        # Le indicamos con la función argsort que ordene nuesta poblacion, pero como ordena de menos a mayor
        # agregamos el [::1] para que invierta el orden

        organized_population = expanded_population[indices] 
        # Matriz apartir de los indices que ya tenemos

        select_organized_population = organized_population[0:n, :]
        # Seleccionamos hasta el limite establecido "n"

        cleaned_population = select_organized_population[:,0:c]
        # Quitamos el indice de aptitud para quedarnos solo con la población

        for i in range(r):
            for j in range(c):
                if (i < n):
                    pnew[i,j] = cleaned_population[i,j]
                    # Insertamos la población nueva a la matriz de selección
                else:
                    pnew[i, j] = np.random.randint(0,2)
                    # Cuando se acaba, rellenamos con numeros aleatorios
        return pnew
