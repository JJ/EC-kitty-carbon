from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
from examples.treegp.non_sklearn_mode.symbolic_regression.sym_reg_evaluator import SymbolicRegressionEvaluator

from codecarbon import track_emissions
from pyJoules.energy_meter import measure_energy

@measure_energy
def main():
    algo = SimpleEvolution(Subpopulation(SymbolicRegressionEvaluator()))
    algo.evolve()
    print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')