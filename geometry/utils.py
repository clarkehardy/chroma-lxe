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

def cylinder(begin, radius, length, direction):
    import trimesh

    # create a cylinder
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)
    cylinder.vertices[:, 2] -= length / 2
    # move the cylinder to the beginning
    # rotate the cylinder to the direction
    rotmat = gen_rot([0, 0, 1], direction)
    transformation_mx = np.eye(4)
    transformation_mx[:3, :3] = rotmat
    cylinder.apply_transform(transformation_mx)

    cylinder.apply_translation(begin)
    return cylinder

def wvl2rgb(wvl):
    if wvl < 380 or wvl > 780:
        raise ValueError("Wavelength must be between 380 and 780 nm")
    if wvl < 440:
        r = -(wvl - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wvl < 490:
        r = 0.0
        g = (wvl - 440) / (490 - 440)
        b = 1.0
    elif wvl < 510:
        r = 0.0
        g = 1.0
        b = -(wvl - 510) / (510 - 490)
    elif wvl < 580:
        r = (wvl - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wvl < 645:
        r = 1.0
        g = -(wvl - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0
    return np.array([r, g, b])

def um2mm(x):
    """Micron to millimeter conversion"""
    return x / 1e3

def mm2um(x):
    """Millimeter to micron conversion"""
    return x * 1e3


