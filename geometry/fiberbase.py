import numpy as np
from chroma.event import Photons
from chroma.sample import uniform_sphere
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Tuple, List, Literal
from scipy.interpolate import interp1d
from scipy.special import jn

from utils.mesh import cylinder
from utils.mesh import um2mm, mm2um
from utils.mesh import gen_rot

class FiberMeta(type(BaseModel)):
    def __new__(mcs, name, bases, namespace):
        annotations = namespace.get("__annotations__", {})
        for key, value in namespace.items():
            if not key.startswith("__") and key not in annotations:
                if isinstance(value, (int, float)):
                    annotations[key] = float
                elif isinstance(value, np.ndarray):
                    annotations[key] = np.ndarray
                elif isinstance(value, str):
                    annotations[key] = Literal[value] # type: ignore
                elif isinstance(value, tuple) and len(value) == 3:
                    annotations[key] = Tuple[float, float, float]
        namespace["__annotations__"] = annotations
        return super().__new__(mcs, name, bases, namespace)


class BaseFiber(BaseModel, metaclass=FiberMeta):
    
    # nanometers
    wavelength: np.ndarray = Field(...)
    
    # arbitrary units
    intensity: np.ndarray = Field(...)
    
    # micrometers
    diameter: float = Field(...)

    numerical_aperture: float = Field(...)
    mode: Literal["cladding", "gaussian"] = "cladding"
    
    # millimeters
    position: Tuple[float, float, float] = (0, 0, 0)
    direction: Tuple[float, float, float] = (0, 0, 1)

    class Config:
        arbitrary_types_allowed = True

    @field_validator("direction")
    def normalize_direction(cls, v):
        return tuple(np.array(v) / np.linalg.norm(v))

    @model_validator(mode="after")
    def check_wavelength_intensity_length(self):
        assert len(self.wavelength) == len(
            self.intensity
        ), "Wavelength and intensity arrays must have the same length"
        return self

    def __init__(self, **data):
        # Load wavelength and intensity from class attributes if not provided
        if "wavelength" not in data and hasattr(self.__class__, "wavelength"):
            data["wavelength"] = self.__class__.wavelength
        if "intensity" not in data and hasattr(self.__class__, "intensity"):
            data["intensity"] = self.__class__.intensity

        super().__init__(**data)
        self._initialize_properties()

    @property
    def cdf(self):
        if not hasattr(self.__class__, "_cdf"):
            _cdf = np.cumsum(self.intensity)
            _cdf /= _cdf[-1]
            self._cdf = _cdf
        return self._cdf

    @property
    def rotation_matrix(self):
        if not hasattr(self.__class__, "_rotation_matrix"):
            _rotation_matrix = gen_rot([0, 0, 1], self.direction)
            self._rotation_matrix = _rotation_matrix
        return self._rotation_matrix

    def _initialize_properties(self):
        self.diameter = um2mm(self.diameter)  # chroma uses mm
        self.direction = np.array(self.direction)
        self.position = np.array(self.position)

    def wavelength_sampler(self, num_samples: int) -> np.ndarray:
        random_values = np.random.rand(num_samples)
        sampled_wavelengths = np.interp(random_values, self.cdf, self.wavelength)
        return sampled_wavelengths

    def sample_positions(self, num_samples: int) -> np.ndarray:        
        if self.mode == "cladding":
            return self._sample_positions_cladding(num_samples)
        else:
            return self._sample_positions_gaussian(num_samples)

    def _sample_positions_gaussian(self, num_samples: int) -> np.ndarray:
        """Sample positions from a radially symmetric Gaussian distribution via Box-Muller method

        See: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        """
        
        w_0 = self.diameter / 2
        U = np.random.rand(num_samples)
        r = w_0 * np.sqrt(-np.log(U) / 2)
        r = r[r <= w_0]

        theta = 2 * np.pi * np.random.rand(len(r))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(len(r))
        pos = np.stack((x, y, z), axis=-1)

        def transform(x):
            return np.dot(x, self.rotation_matrix.T) \
            + self.position \
            + 1e-3 * self.direction

        while len(pos) < num_samples:
            return np.vstack(
                (
                    transform(pos),
                    self._sample_positions_gaussian(num_samples - len(pos)),
                )
            )

        return transform(pos)

    def _sample_positions_cladding(self, num_samples: int) -> np.ndarray:
        """Sample positions from an intensity distribution that emperically looks
        like a cladding mode.
        
        We use the Bessel function of the first kind of order 5 to generate the intensity
        distribution. The radial positions are sampled by inverting the cumulative
        distribution function of the intensity distribution.
        """
        
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

        def transform(x):
            return (
                np.dot(x, self.rotation_matrix.T)
                + self.position
                + 1e-3 * self.direction
            )

        return transform(pos)

    def direction_sampler(self, num_samples: int) -> np.ndarray:
        """Sample directions from a uniform distribution within the numerical aperture of the fiber."""
        
        divergence_angle = np.arcsin(self.numerical_aperture)
        phi = 2 * np.pi * np.random.rand(num_samples)
        theta = divergence_angle * np.sqrt(np.random.rand(num_samples))
        directions = np.stack(
            (np.sin(theta) * np.cos(phi), \
             np.sin(theta) * np.sin(phi), \
             np.cos(theta)),
            axis=-1,
        )
        dirr = np.dot(directions, self.rotation_matrix.T)
        print('sampled direction pre-rotation\n', directions[0])
        print('sampled direction post-rotation\n', dirr[0])
        return np.dot(directions, self.rotation_matrix.T)

    def generate_photons(self, num_photons: int) -> Photons:
        """Generate photons with random positions, directions, polarizations, and wavelengths.
        
        Directions are sampled from a uniform distribution within the numerical aperture of the fiber.
        Polarizations are generated by taking the cross product of the direction and a random vector.
        Positions are sampled from a radially symmetric Gaussian distribution or an empirical cladding mode.
        Wavelengths are sampled from the intensity distribution of the fiber.
        """

        num_photons = int(num_photons)
        positions = self.sample_positions(num_photons)
        directions = self.direction_sampler(num_photons)
        polarizations = np.cross(directions, uniform_sphere(num_photons))
        initial_wavelengths = self.wavelength_sampler(num_photons)
        return Photons(positions, directions, polarizations, initial_wavelengths)

    def generate_photons_mesh(self, num_photons: int = 1000, concatenate: bool = True) -> List:
        """Generate photons and return them as a mesh of cylinders to quickly visualize them in trimesh.
        
        Usage:
        ```python
        fiber = M114L01()
        mesh = fiber.generate_photons_mesh(1000)
        mesh.show()
        ```

        """
        
        import trimesh
        from matplotlib.colors import LogNorm
        from matplotlib.cm import inferno

        phots = self.generate_photons(num_photons)        
        colors = inferno(LogNorm()(phots.wavelengths))
        
        ortho = np.cross(self.direction, uniform_sphere())
        ortho /= np.linalg.norm(ortho)
        axis = trimesh.creation.axis(
            origin_size=0.3,
            axis_length=2,
            transform=trimesh.transformations.translation_matrix(
                self.position + 2 * ortho
            ),
        )
        # get an orthogonal vector to the direction
        print('note: XYZ->RGB')        

        cyls = []
        for i, (pos, dir) in enumerate(zip(phots.pos, phots.dir)):
            cyl = cylinder(pos, 0.01, 20, dir)
            cyl.visual.face_colors = colors[i]
            cyls.append(cyl)
        return trimesh.util.concatenate(cyls) + axis if concatenate else cyls + [axis]

def main():
    
    class DummyFiber(BaseFiber):
        wavelength = np.linspace(380, 780, 1000)
        intensity = np.random.rand(1000)
        diameter = 600
        numerical_aperture = 0.22
        mode = "cladding"
    
    fiber = DummyFiber()
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
    radii = mm2um(np.linalg.norm(photons.pos[:, :2], axis=1))
    print("\tavg. radius:", radii.mean(), "um")
    print("\tmax. radius:", radii.max(), "um")
    print("\tmin. radius:", radii.min(), "um")



if __name__ == "__main__":
    main()
