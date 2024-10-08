import numpy as np

def vprint(verb_setting, verb, *args):
    # global verb_setting
    if verb <= verb_setting:
        print(*args)


def sind(alpha):
    return np.sin(alpha * np.pi / 180.0)


def cosd(alpha):
    return np.cos(alpha * np.pi / 180.0)


def norm2(x):
    return np.linalg.norm(x, 2)


def atan2(y, x):
    return np.arctan2(y, x)
