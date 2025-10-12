import numpy as np
import pandas as pd


# *****************************Class of algoritm cuantitative for rank***********************************
class cl_alg_quantum():
    def __init__(self, funtion, population, num_qubits, cant_ciclos,
                 mutation_percent, i_min,
                    i_max, optimum):
        self.funtion = funtion
        self.population = population
        self.num_qubits = num_qubits
        self.cant_ciclos = cant_ciclos
        self.mutation_percent = mutation_percent
        self.i_min = i_min
        self.i_max = i_max
        self.optimum = optimum


    def run(self):
        print(f"\n[INFO] Starting Quantum algorithm")
        self.quantum_amplitudes_population = self.quantum_amplitudes(self.population, self.num_qubits)
        self.best_solution = None
        self.best_fitness = float('-inf')

        for i in range(self.cant_ciclos):
            self.solutions = self.generate_solutions(self.quantum_amplitudes_population)
            self.fitness_population = self.fitness_quantum_population(self.solutions, self.i_min, self.i_max)

            max_fitness_index = np.argmax(self.fitness_population)
            if self.fitness_population[max_fitness_index] > self.best_fitness:
                self.best_fitness = self.fitness_population[max_fitness_index]
                self.best_solution = self.solutions[max_fitness_index]

            self.quantum_amplitudes_population = self.rotacionGate(self.quantum_amplitudes_population, self.solutions, self.best_solution)
            self.quantum_amplitudes_population = self.mutations_quantum_population(self.quantum_amplitudes_population, self.mutation_percent)
        
        best_population_decimal = self.decode_quantum_population(self.best_solution.reshape(1, -1), self.i_min, self.i_max)
        best_individual = best_population_decimal[0] 
        print(f"[INFO] Best solution found: {best_individual} with fitness: {self.best_fitness}")
        print(f"[INFO] Ending Quantum algorithm\n")   
        return best_individual


    def quantum_amplitudes(self, population, num_qubits):
        amplitudes = np.full((population, num_qubits), 1 / np.sqrt(2))
        return amplitudes

    # Generate solutions to evaluate the quantum population to collapse the qubits into binary values
    def generate_solutions(self, amplitudes):
            [r, c] = amplitudes.shape
            matrix_binary = np.zeros((r,c), dtype=int)
            matrix_random = np.random.rand(r,c)
            
            # Collapse the qubits
            for i in range(r):
                for j in range(c):
                    if matrix_random[i,j] < amplitudes[i,j]:
                        matrix_binary[i,j] = 1
                    else:
                        matrix_binary[i, j] = 0
            
            return matrix_binary
    
    def decode_quantum_population(self, P, Imin, Imax):
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

    def fitness_quantum_population(self, population, Imin, Imax):
        x = self.decode_quantum_population(population, Imin, Imax)
        fitness = self.funtion(x)
        return fitness
    

    # Evolution operators for quantum algorithm
    def rotacionGate(self, P, soluciones, mejorSolucion, theta = 0.1):
        [r, c] = P.shape
        
        for i in range(r):
            for j in range(c):
                if soluciones[i, j] != mejorSolucion[j]:
                    if mejorSolucion[j] == 1:
                        P[i, j] = P[i, j] + np.sin(theta) * (1 - P[i, j])
                    else:
                        P[i, j] = P[i, j] - np.sin(theta) * P[i, j]
        
        P = np.clip(P, 0.05, 0.95)
        return P
    
    def mutations_quantum_population(self, population, mutation_percent):
        [r, c] = population.shape
        matrixAux = np.random.rand(r,c)
        mascara = np.zeros((r, c), dtype=bool)

        for i in range(r):
            for j in range(c):
                if matrixAux[i,j] < mutation_percent:
                    mascara[i,j] = True
                else:
                    mascara[i,j] = False
        
        population = np.where(mascara, 1 - population, population)
        return population

    



