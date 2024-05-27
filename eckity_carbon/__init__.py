from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
from examples.treegp.non_sklearn_mode.symbolic_regression.sym_reg_evaluator import SymbolicRegressionEvaluator

from codecarbon import track_emissions
@track_emissions(offline=True, country_iso_code="ESP")
def main():
    algo = SimpleEvolution(Subpopulation(SymbolicRegressionEvaluator()))
    algo.evolve()
    print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')