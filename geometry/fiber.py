import pickle

import numpy as np
from chroma.event import Photons
from chroma.sample import uniform_sphere
from scipy.interpolate import interp1d
from scipy.special import jn
try:
    from . import utils
except:
    import utils


class Fiber:
    def __init__(
        self,
        wavelength,
        intensity,
        diameter,
        numerical_aperture,
        mode="cladding",
        position=(0, 0, 0),
        direction=(0, 0, 1),
    ):
        """
        Initialize a Fiber object with the given parameters.

        Parameters:
        - wavelength (float or array-like): The wavelength(s) of the fiber.
        - intensity (array-like): The intensity values corresponding to each wavelength.
        - diameter (float): The diameter of the fiber, in microns.
        - numerical_aperture (float): The numerical aperture of the fiber.
        - mode (str, optional): The intensity mode of the fiber. Can be either 'cladding' or 'gaussian'. Defaults to 'cladding'.

        Raises:
        - AssertionError: If the wavelength and intensity arrays have different lengths, if the diameter is not positive,
            if the numerical aperture is not between 0 and 1, or if the profile is not 'cladding' or 'gaussian'.

        """
        if isinstance(wavelength, float):
            self.wavelength = np.array([wavelength])
            self.intensity = np.array([1])

        assert len(wavelength) == len(
            intensity
        ), "Wavelength and intensity arrays must have the same length"
        assert diameter > 0, "Fiber diameter must be positive"
        assert 0 < numerical_aperture <= 1, "Numerical aperture must be between 0 and 1"
        assert mode in [
            "cladding",
            "gaussian",
        ], "Mode must be either 'cladding' or 'gaussian'"

        self.wavelength = wavelength
        self.cdf = np.cumsum(intensity)
        self.cdf /= self.cdf[-1]

        self.diameter = um2mm(diameter)  # chroma uses mm
        self.numerical_aperture = numerical_aperture
        if mode == "cladding":
            self.position_sampler = self.sample_positions_cladding
        else:
            self.position_sampler = self.sample_positions_gaussian

        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.rotation_matrix = utils.gen_rot([0, 0, -1], self.direction)

    def wavelength_sampler(self, num_samples):
        random_values = np.random.rand(num_samples)
        sampled_wavelengths = np.interp(random_values, self.cdf, self.wavelength)
        return sampled_wavelengths

    def sample_positions_gaussian(self, num_samples):
        w_0 = self.diameter / 2
        U = np.random.rand(num_samples)
        r = w_0 * np.sqrt(-np.log(U) / 2)
        r = r[r <= w_0]

        theta = 2 * np.pi * np.random.rand(len(r))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(len(r))
        pos = np.stack((x, y, z), axis=-1)

        while len(pos) < num_samples:
            return np.dot(
                np.vstack(
                (pos, self.sample_positions_gaussian(num_samples - len(pos)))
            ),
                self.rotation_matrix.T,
            ) + self.position + 1e-3 * self.direction

        return (
            np.dot(pos, self.rotation_matrix.T) + self.position + 1e-3 * self.direction
        )

    def sample_positions_cladding(self, num_samples):
        r = np.linspace(0, self.diameter / 2, 500)
        intensity = np.abs(jn(5, r * 8.5 / (self.diameter / 2))) ** 2
        cdf_intensity = np.cumsum(intensity)
        cdf_intensity /= cdf_intensity[-1]

        random_values = np.random.rand(num_samples)
        interp_r = interp1d(
            cdf_intensity, r, fill_value="extrapolate", bounds_error=False
        )
        sampled_r = interp_r(random_values)

        theta = 2 * np.pi * np.random.rand(num_samples)
        x = sampled_r * np.cos(theta)
        y = sampled_r * np.sin(theta)
        z = np.zeros(num_samples)
        pos = np.stack((x, y, z), axis=-1)

        while len(pos) < num_samples:
            return (
                np.dot(
                    np.vstack(
                        (pos, self.sample_positions_cladding(num_samples - len(pos)))
                    ),
                    self.rotation_matrix.T,
                )
                + self.position
                + 1e-3 * self.direction
            )

        return np.dot(pos, self.rotation_matrix.T) + self.position + 1e-3 * self.direction

    def direction_sampler(self, num_samples):
        # Sample initial directions based on the numerical aperture
        divergence_angle = np.arcsin(self.numerical_aperture)

        # Generate random angles for direction vectors within the allowed divergence cone
        phi = 2 * np.pi * np.random.rand(num_samples)  # Azimuthal angle
        theta = divergence_angle * np.sqrt(
            np.random.rand(num_samples)
        )  # Polar angle scaled by divergence angle

        # Convert spherical coordinates to Cartesian coordinates for direction
        directions = np.stack(
            (np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)),
            axis=-1,
        )

        # Normalize directions to unit vectors
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        return np.dot(directions, self.rotation_matrix.T)

    def generate_photons(self, num_photons):
        num_photons = int(num_photons)
        positions = self.position_sampler(num_photons)
        directions = self.direction_sampler(num_photons)
        polarizations = np.cross(directions, uniform_sphere(num_photons))
        initial_wavelengths = self.wavelength_sampler(num_photons)

        return Photons(positions, directions, polarizations, initial_wavelengths)
    
    def generate_photons_mesh(self, num_photons=1000):
        import trimesh
        
        phots = self.generate_photons(num_photons)
        # pointcloud
        pc = trimesh.points.PointCloud(phots.pos)
        # set color to red
        pc.colors = list(
            map(utils.wvl2rgb, np.clip(phots.wavelengths - phots.wavelengths.min() + 380, 380, 780))
        )

        cyls = []
        for i, (pos, dir) in enumerate(zip(phots.pos, phots.dir)):
            cyl = utils.cylinder(pos, 0.01, 20, dir)
            cyl.visual.face_colors = pc.colors[i]

            cyls.append(cyl)
        return cyls

class M114L01(Fiber):
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

    numerical_aperture = 0.22
    diameter = 600  # um
    mode = "cladding"

    def __init__(self, position=(0, 0, 0), direction=(0, 0, 1), mode='cladding'):
        xe_wvl, xe_intensities = pickle.load(
            open("/home/sam/sw/chroma-lxe/data/xe-spectrum.p", "rb")
        )
        super().__init__(
            xe_wvl,
            xe_intensities,
            self.diameter,
            self.numerical_aperture,
            mode,
            position,
            direction,
        )


def um2mm(x):
    return x / 1e3


def mm2um(x):
    return x * 1e3


def main():
    fiber = M114L01()
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
    print(
        "\tavg. radius:",
        mm2um(np.mean(np.linalg.norm(photons.pos[:, :2], axis=1))),
        "um",
    )
    print(
        "\tmax. radius:",
        mm2um(np.max(np.linalg.norm(photons.pos[:, :2], axis=1))),
        "um",
    )
    print(
        "\tmin. radius:",
        mm2um(np.min(np.linalg.norm(photons.pos[:, :2], axis=1))),
        "um",
    )


if __name__ == "__main__":
    main()
