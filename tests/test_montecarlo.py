import ast
from pathlib import Path
import numpy as np


def load_run_monte_carlo_simulations():
    path = Path(__file__).resolve().parents[1] / "montecarlo.py"
    source = path.read_text()
    module = ast.parse(source)
    func_node = None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_monte_carlo_simulations":
            func_node = node
            break
    if func_node is None:
        raise AssertionError("Function run_monte_carlo_simulations not found")
    func_node.decorator_list = []  # Remove streamlit decorator
    func_module = ast.Module(body=[func_node], type_ignores=[])
    code_obj = compile(func_module, filename=str(path), mode="exec")
    namespace = {"np": np}
    exec(code_obj, namespace)
    return namespace["run_monte_carlo_simulations"]


def test_output_shape():
    func = load_run_monte_carlo_simulations()
    np.random.seed(0)
    mean_returns = np.array([0.01, 0.02])
    cov_matrix = np.array([[0.1, 0.02], [0.02, 0.1]])
    results = func(
        num_simulations=5,
        time_horizon=10,
        weights=[0.5, 0.5],
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        initial_value=1000,
    )
    assert results.shape == (5, 10)


def test_reproducibility():
    func = load_run_monte_carlo_simulations()
    mean_returns = np.array([0.0, 0.0])
    cov_matrix = np.array([[0.05, 0.0], [0.0, 0.05]])

    np.random.seed(42)
    out1 = func(
        num_simulations=3,
        time_horizon=4,
        weights=[0.6, 0.4],
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        initial_value=1000,
    )

    np.random.seed(42)
    out2 = func(
        num_simulations=3,
        time_horizon=4,
        weights=[0.6, 0.4],
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        initial_value=1000,
    )

    np.testing.assert_allclose(out1, out2)
