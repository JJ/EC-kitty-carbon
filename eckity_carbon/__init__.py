from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.subpopulation import Subpopulation
from examples.treegp.non_sklearn_mode.symbolic_regression.sym_reg_evaluator import SymbolicRegressionEvaluator

from ea2p import PowerMeter


def main():
    algo = SimpleEvolution(Subpopulation(SymbolicRegressionEvaluator()))
    with PowerMeter() as meter:
        algo.evolve()
    print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')