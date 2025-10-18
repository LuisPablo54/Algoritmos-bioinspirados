from .gen import GEN
from .algorithms.alg_bin import cl_alg_stn_bin_rank
from .algorithms.alg_quantum import cl_alg_quantum

class RelaxGEN(GEN):
    def __init__(self, funtion=None, population=None, **kwargs):
        super().__init__(funtion, population, **kwargs)
        # Almacena los parámetros específicos como atributos directos
        self.cant_genes = kwargs.get("cant_genes")
        self.num_cycles = kwargs.get("num_cycles")
        self.selection_percent = kwargs.get("selection_percent")
        self.crossing = kwargs.get("crossing")
        self.mutation_percent = kwargs.get("mutation_percent")
        self.i_min = kwargs.get("i_min")
        self.i_max = kwargs.get("i_max")
        self.optimum = kwargs.get("optimum")
        self.num_qubits = kwargs.get("num_qubits")
    
    def alg_stn_bin_rank(self):
        algoritmo = cl_alg_stn_bin_rank(
            funtion=self.funtion,
            population=self.population,
            cant_genes=self.cant_genes,
            cant_ciclos=self.num_cycles,
            selection_percent=self.selection_percent,
            crossing=self.crossing,
            mutation_percent=self.mutation_percent,
            i_min=self.i_min,
            i_max=self.i_max,
            optimum=self.optimum
        )
        return algoritmo.run()

    def alg_quantum(self):
        algoritmo = cl_alg_quantum(
            funtion=self.funtion,
            population=self.population,
            num_qubits=self.num_qubits,
            cant_ciclos=self.num_cycles,
            mutation_percent=self.mutation_percent,
            i_min=self.i_min,
            i_max=self.i_max,
            optimum=self.optimum
        )
        return algoritmo.run()
