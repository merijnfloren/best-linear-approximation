import numpy as np

from best_linear_approximation._linear_algebra import kronecker_product, right_solve


def test_kronecker_product_matches_numpy_for_matrices() -> None:
    rng = np.random.default_rng(0)
    left = rng.random((2, 3))
    right = rng.random((4, 5))

    result = kronecker_product(left, right)

    np.testing.assert_allclose(result, np.kron(left, right))


def test_right_solve_solves_known_complex_system() -> None:
    right = np.array([[2.0 + 1.0j, 1.0j], [1.0, 3.0 - 2.0j]])
    known_solution = np.array([[1.0 + 2.0j, -2.0j], [3.0, 4.0 - 1.0j]])
    left = known_solution @ right

    inferred_solution = right_solve(left, right)

    np.testing.assert_allclose(inferred_solution, known_solution)


def test_right_solve_broadcasts_leading_axes() -> None:
    right = np.broadcast_to(
        np.array([[2.0, 1.0], [1.0, 3.0]]),
        (1, 3, 2, 2),
    )
    known_solution = np.arange(8, dtype=complex).reshape(2, 1, 2, 2)
    left = known_solution @ right

    inferred_solution = right_solve(left, right)

    np.testing.assert_allclose(
        inferred_solution,
        np.broadcast_to(known_solution, left.shape),
    )
