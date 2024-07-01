import numpy as np
from chroma.sample import uniform_sphere
from chroma.event import Photons


def create_photon_bomb(n: int, wavelength: float, pos: np.ndarray) -> Photons:
    pos = np.tile(pos, (n, 1))
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(wavelength, n)
    return Photons(pos, dir, pol, wavelengths)
