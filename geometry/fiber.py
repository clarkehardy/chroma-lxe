import pickle

import numpy as np
from chroma.event import Photons
from scipy.interpolate import interp1d
from scipy.special import jn
import utils


class Fiber:
    def __init__(
        self,
        wavelength,
        intensity,
        diameter,
        numerical_aperture,
        profile="cladding",
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
        - profile (str, optional): The profile of the fiber. Can be either 'cladding' or 'gaussian'. Defaults to 'cladding'.

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
        assert profile in [
            "cladding",
            "gaussian",
        ], "Profile must be either 'cladding' or 'gaussian'"

        self.wavelength = wavelength
        self.cdf = np.cumsum(intensity)
        self.cdf /= self.cdf[-1]

        self.diameter = um2mm(diameter)  # chroma uses mm
        self.numerical_aperture = numerical_aperture
        if profile == "cladding":
            self.position_sampler = self.sample_positions_cladding
        else:
            self.position_sampler = self.sample_positions_gaussian

        self.position = np.array(position)
        direction = np.array(direction) / np.linalg.norm(direction)
        self.rotation_matrix = utils.gen_rot([0, 0, 1], direction)

    def sample_wavelengths(self, num_samples):
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
            return np.vstack(
                (pos, self.sample_positions_gaussian(num_samples - len(pos)))
            )

        return pos

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

        print(sampled_r.max(), sampled_r.min(), self.diameter / 2)

        theta = 2 * np.pi * np.random.rand(num_samples)
        x = sampled_r * np.cos(theta)
        y = sampled_r * np.sin(theta)
        z = np.zeros(num_samples)
        pos = np.stack((x, y, z), axis=-1)

        while len(pos) < num_samples:
            return np.vstack(
                (pos, self.sample_positions_cladding(num_samples - len(pos)))
            )

        return pos

    def sample_directions(self, num_samples):
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
        return directions

    def generate_photons(self, num_photons):
        num_photons = int(num_photons)
        positions = self.position_sampler(num_photons)
        directions = self.sample_directions(num_photons)
        initial_wavelengths = self.sample_wavelengths(num_photons)

        return Photons(positions, directions, wavelengths=initial_wavelengths)


class M114L01(Fiber):
    def __init__(self, position=(0, 0, 0), direction=(0, 0, 1)):
        xe_wvl, xe_intensities = pickle.load(
            open("/home/sam/sw/chroma-lxe/data/xe-spectrum.p", "rb")
        )
        NA = 0.22
        diameter = 600  # um
        profile = "cladding"
        super().__init__(
            xe_wvl, xe_intensities, diameter, NA, profile, position, direction
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