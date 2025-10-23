from utils.math_utils import round_step


def test_round_step_floor():
    assert round_step(1.2345, 0.01, "floor") == 1.23


def test_round_step_ceil():
    assert round_step(1.2301, 0.01, "ceil") == 1.24


def test_round_step_round():
    assert round_step(1.235, 0.01, "round") == 1.24
