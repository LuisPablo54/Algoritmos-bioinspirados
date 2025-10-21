from .gen import GEN
from .algorithms.alg_bin import cl_alg_stn_bin
from .algorithms.alg_quantum import cl_alg_quantum

class RelaxGEN(GEN):
    def __init__(self, funtion=None, population=None, **kwargs):
        # --- Defaults si no se pasan ---
        defaults = {
            "cant_genes": 8,
            "num_cycles": 100,
            "selection_percent": 0.5,
            "crossing": 0.5,
            "mutation_percent": 0.3,
            "i_min": -10,
            "i_max": 10,
            "optimum": "max",
            "num_qubits": 8,
            "select_mode": "ranking",
        }

        # Sobrescribe defaults con los valores dados por el usuario
        defaults.update(kwargs)

        # Llama al constructor de la clase base GEN
        super().__init__(funtion, population, **defaults)

        # Almacena los parámetros finales (ya con defaults aplicados)
        self.cant_genes = defaults["cant_genes"]
        self.num_cycles = defaults["num_cycles"]
        self.selection_percent = defaults["selection_percent"]
        self.crossing = defaults["crossing"]
        self.mutation_percent = defaults["mutation_percent"]
        self.i_min = defaults["i_min"]
        self.i_max = defaults["i_max"]
        self.optimum = defaults["optimum"]
        self.num_qubits = defaults["num_qubits"]
        self.select_mode = defaults["select_mode"]

    def alg_stn_bin(self):
        algoritmo = cl_alg_stn_bin(
            funtion=self.funtion,
            population=self.population,
            cant_genes=self.cant_genes,
            cant_ciclos=self.num_cycles,
            selection_percent=self.selection_percent,
            crossing=self.crossing,
            mutation_percent=self.mutation_percent,
            i_min=self.i_min,
            i_max=self.i_max,
            optimum=self.optimum,
            select_mode=self.select_mode
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
