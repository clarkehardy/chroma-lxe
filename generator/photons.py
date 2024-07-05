import numpy as np
from chroma.sample import uniform_sphere
from chroma.event import Photons


def create_photon_bomb(n: int, wavelength: float, pos: np.ndarray) -> Photons:
    """Create a collection of photons at a given position with random directions.
    
    Parameters
    ----------
    n : int
        The number of photons to create.
    wavelength : float
        The wavelength of the photons.
    pos : array-like
        The position of the photons.
        
    Returns
    -------
    photons : chroma.event.Photons
        The collection of photons

    """

    pos = np.tile(pos, (n, 1))
    dir = uniform_sphere(n)
    pol = np.cross(dir, uniform_sphere(n))
    wavelengths = np.repeat(wavelength, n)
    return Photons(pos, dir, pol, wavelengths)
