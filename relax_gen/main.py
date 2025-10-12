'''
Description:

En este archivo se econtraran diferentes modelos genéticos, de manera que se puedan utilizar en el proyecto
de manera sencilla y rápida.

Se agregarán más modelos conforme se vayan necesitando.
'''

import numpy as np
import pandas as pd

# Import the algorithms to be used
from algritmos.alg_bin import cl_alg_stn_bin_rank
from algritmos.alg_quantum import cl_alg_quantum


class GEN():
    def __init__(self, funtion, population, cant_genes = None, num_cycles= None, selection_percent = None, 
                 crossing = None, mutation_percent = None, i_min = None, i_max = None, optimum = None, num_qubits = None):
        self.funtion = funtion
        self.population = population
        self.cant_genes = cant_genes
        self.num_qubits = num_qubits
        self.cant_ciclos = num_cycles
        self.selection_percent = selection_percent
        self.crossing = crossing
        self.mutation_percent = mutation_percent
        self.i_min = i_min
        self.i_max = i_max
        self.optimum = optimum 

    def alg_stn_bin_rank(self):
        algoritmo = cl_alg_stn_bin_rank(
            self.funtion,
            self.population,
            self.cant_genes,
            self.cant_ciclos,
            self.selection_percent,
            self.crossing,
            self.mutation_percent,
            self.i_min,
            self.i_max,
            self.optimum
        )
        return algoritmo.run()

    def alg_quantum(self):
        algoritmo = cl_alg_quantum(
            self.funtion,
            self.population,
            self.num_qubits,
            self.cant_ciclos,
            self.mutation_percent,
            self.i_min,
            self.i_max,
            self.optimum
        )
        return algoritmo.run()


