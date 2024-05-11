import os
from typing import Optional

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from src.common import NDArrayFloat
from src.linalg import get_scipy_solution
from practicum_8.conjugate_gradient_method import (
    conjugate_gradient_descent,
    relative_error,
)


def iterative_refinement(
    A: NDArrayFloat, b: NDArrayFloat, solver, n_iters: int, n_ir_iters: int
) -> NDArrayFloat:

    ##########################
    ### PUT YOUR CODE HERE ###
    ##########################

    pass


def add_convergence_graph_to_axis(
    ax, exact_solution: NDArrayFloat, solution_history: NDArrayFloat
) -> None:
    n_iters = solution_history.shape[0]
    ax.semilogy(
        range(n_iters),
        relative_error(x_true=exact_solution, x_approx=solution_history),
        "o--",
    )
    ax.grid()
    ax.legend(fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$||x - \tilde{x}|| / ||x||$", fontsize=12)


if __name__ == "__main__":
    np.random.seed(42)

    # Download s1rmq4m1.mtx.gz here:
    # https://math.nist.gov/MatrixMarket/data/misc/cylshell/s1rmq4m1.html

    path_to_matrix = os.path.join(
        "practicum_6", "homework", "advanced", "matrices", "s1rmq4m1.mtx.gz"
    )
    A = scipy.io.mmread(path_to_matrix).todense().A
    b = np.ones((A.shape[0],))
    exact_solution = get_scipy_solution(A, b)
    n_iters = 1000
    n_ir_iters = 5

    dtype = np.float32
    A = A.astype(np.float32)
    b = b.astype(np.float32)

    # Convergence speed for the conjugate gradient method
    ir_solution_history = iterative_refinement(
        A, b, solver=conjugate_gradient_descent, n_iters=n_iters, n_ir_iters=n_ir_iters
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    add_convergence_graph_to_axis(ax, exact_solution, ir_solution_history)
    plt.show()
