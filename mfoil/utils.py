import numpy as np

# ============ INPUT, OUTPUT, UTILITY ==============

# -------------------------------------------------------------------------------
def vprint(param, verb, *args):
    if verb <= param.verb:
        print(*args)


# -------------------------------------------------------------------------------
def sind(alpha):
    return np.sin(alpha * np.pi / 180.0)


# -------------------------------------------------------------------------------
def cosd(alpha):
    return np.cos(alpha * np.pi / 180.0)


# -------------------------------------------------------------------------------
def norm2(x):
    return np.linalg.norm(x, 2)


# -------------------------------------------------------------------------------
def dist(a, b):
    return np.sqrt(a**2 + b**2)


# -------------------------------------------------------------------------------
def atan2(y, x):
    return np.arctan2(y, x)