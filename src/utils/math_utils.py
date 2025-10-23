import math


def round_step(v: float, step: float, how: str = "floor") -> float:
    if not step:
        return v
    n = v / step
    if how == "floor":
        n = math.floor(n + 1e-12)
    elif how == "ceil":
        n = math.ceil(n - 1e-12)
    else:
        n = round(n)
    return n * step
