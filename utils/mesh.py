import numpy as np
from chroma.transform import make_rotation_matrix


# def gen_rot(a, b):
#     """Generate a rotation matrix that rotates vector a to vector b"""
#     a,b = b,a
#     a = a / np.linalg.norm(a)
#     b = b / np.linalg.norm(b)

#     if np.allclose(a, b):
#         return np.eye(3)
#     elif np.allclose(a, -b):
#         if a[1] == 0 and a[2] == 0:
#             v = np.cross(a, [0, 1, 0])
#         else:
#             v = np.cross(a, [1, 0, 0])
#         c = np.pi
#     else:
#         v = np.cross(a, b)
#         c = np.arccos(np.dot(a, b))
#     return make_rotation_matrix(c, v)

def gen_rot(a,b):
    a = np.array(a)
    b = np.array(b)
    
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if np.isclose(c, 1):
        # Vectors are already aligned
        return np.identity(3)
    elif np.isclose(c, -1):
        # Vectors are anti-parallel
        # Find orthogonal vector to a to form a valid rotation axis
        orthogonal_vector = np.array([1, 0, 0])
        if np.allclose(a, orthogonal_vector):
            orthogonal_vector = np.array([0, 1, 0])
        v = np.cross(a, orthogonal_vector)
        v = v / np.linalg.norm(v)
        v_skew = np.array([[    0, -v[2],  v[1]],
                           [ v[2],     0, -v[0]],
                           [-v[1],  v[0],     0]])
        return -np.identity(3) + 2 * np.dot(v[:, None], v[None, :])
    
    s = np.linalg.norm(v)
    v_skew = np.array([[    0, -v[2],  v[1]],
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],     0]])
    I = np.eye(3)
    R = I + v_skew + np.dot(v_skew, v_skew) * ((1 - c) / (s ** 2))
    return R


def cylinder(begin, radius, length, direction):
    """Create a cylinder with a given radius, length, and direction. Return a trimesh object."""
    import trimesh

    # create a cylinder
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)
    cylinder.vertices[:, 2] += length / 2
    # move the cylinder to the beginning
    # rotate the cylinder to the direction
    rotmat = gen_rot([0, 0, 1], direction)
    transformation_mx = np.eye(4)
    transformation_mx[:3, :3] = rotmat
    cylinder.apply_transform(transformation_mx)

    cylinder.apply_translation(begin)
    
    return cylinder

def um2mm(x):
    """Micron to millimeter conversion"""
    return x / 1e3

def mm2um(x):
    """Millimeter to micron conversion"""
    return x * 1e3

