import numpy as np
from chroma.transform import make_rotation_matrix

def gen_rot(a, b):
    """Generate a rotation matrix that rotates vector a to vector b"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    if np.allclose(a, -b):
        return np.eye(3)
    elif np.allclose(a, b):
        if a[1] == 0 and a[2] == 0:
            v = np.cross(a, [0, 1, 0])
        else:
            v = np.cross(a, [1, 0, 0])
        c = np.pi
    else:
        v = np.cross(a, b)
        c = np.arccos(np.dot(a, b))
    return make_rotation_matrix(c, v)
