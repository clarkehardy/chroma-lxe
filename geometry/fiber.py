import pickle

import numpy as np
from fiberbase import BaseFiber


class ExampleFiber(BaseFiber):
    """A class representing an example fiber."""

    wavelength = np.linspace(380, 780, 1000)
    intensity = np.random.rand(1000)
    diameter = 600 # um
    numerical_aperture = 0.22
    mode = "cladding"


class M114L01(BaseFiber):
    """https://www.thorlabs.com/thorproduct.cfm?partnumber=M114

    The mode of this fiber can be a little confusing. This "cladding" mode will
    occur when several fibers are connected in series, resulting in a lot of the
    light being "lost" into the cladding. TIR can still occur in the cladding, so a
    ring-like profile emerges when there's more light stuck in the cladding than the
    core of the fiber. A gaussian-like profile will occur when a single fiber is
    connected to a light source.

    See https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=5591, section "MM
    Fiber Tutorial" for more details.
    """

    diameter = 600
    numerical_aperture = 0.22
    mode = "cladding"
    wavelength, intensity = pickle.load(
        open("/home/sam/sw/chroma-lxe/data/xe-spectrum.p", "rb")
    )
    
def main():
    fiber = M114L01(position=[0, 0, 0], direction=[0, 0, 1])
    photons = fiber.generate_photons(1e5)

    assert np.allclose(
        np.linalg.norm(photons.dir, axis=1), 1
    ), "Directions are not unit vectors"
    assert all(
        np.linalg.norm(photons.pos, axis=1) <= fiber.diameter / 2
    ), "Photons are outside the fiber core"
    assert all(
        np.arccos(np.dot(photons.dir, [0, 0, 1])) <= np.arcsin(fiber.numerical_aperture)
    ), "Photons are outside the numerical aperture"

    print("Fiber data:")
    print("\tdiameter:", fiber.diameter, "mm")
    print("\tnumerical aperture:", fiber.numerical_aperture)
    print("Photon data:")
    print("\tmedian. wavelength:", np.median(photons.wavelengths))
    print("\tavg. wavelength:", np.mean(photons.wavelengths))
    print("\tmax. wavelength:", np.max(photons.wavelengths))
    print("\tmin. wavelength:", np.min(photons.wavelengths))
    radii = np.linalg.norm(photons.pos[:, :2], axis=1)
    print("\tavg. radius:", radii.mean(), "mm")
    print("\tmax. radius:", radii.max(), "mm")
    print("\tmin. radius:", radii.min(), "mm")


if __name__ == "__main__":
    main()
