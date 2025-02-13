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


def create_electroluminescence_photons(n: int, wavelength: float, pos: np.ndarray, height: float) -> Photons:
    """Create a collection of photons at a given position with random directions.
    
    Parameters
    ----------
    n : int
        The number of photons to create.
    wavelength : float
        The wavelength of the photons.
    pos : array-like
        The (x, y, z) position on the liquid surface where electroluminesence begins.
    height : float:
        The height of the electroluminescence region.
        
    Returns
    -------
    photons : chroma.event.Photons
        The collection of photons
    """
    point_spacing = 0.1 # mm
    n_per_point = int(n/np.round(height/point_spacing))
    pos = np.tile(pos, (n, 1))
    z_offsets = np.zeros_like(pos)
    z_points = np.linspace(0, height, int(np.round(height/point_spacing)))
    for i in range(int(np.round(height/point_spacing))):
        z_offsets[i*n_per_point:(i+1)*n_per_point, 2] = z_points[i]
    pos = pos + z_offsets
    dirs = np.zeros_like(pos)
    pols = np.zeros_like(pos)
    wavelengths = np.repeat(wavelength, n)
    for i in range(int(np.round(height/point_spacing))):
        dirs[i*n_per_point:(i+1)*n_per_point] = uniform_sphere(n_per_point)
        pols[i*n_per_point:(i+1)*n_per_point] = np.cross(dirs[i*n_per_point:(i+1)*n_per_point], uniform_sphere(n_per_point))
    return Photons(pos, dirs, pols, wavelengths)


def create_multisite_electroluminescence_photons(n: int, wavelength: float, pos_1: np.ndarray, \
                                                 pos_2: np.ndarray, height: float) -> Photons:
    """Create a collection of photons at a given position with random directions.
    
    Parameters
    ----------
    n : int
        The number of photons to create.
    wavelength : float
        The wavelength of the photons.
    pos : array-like
        The (x, y, z) position on the liquid surface where electroluminesence begins.
    height : float:
        The height of the electroluminescence region.
        
    Returns
    -------
    photons : chroma.event.Photons
        The collection of photons
    """
    n_1 = int(n/2)
    n_2 = n - n_1
    site_1_photons = create_electroluminescence_photons(n_1, wavelength, pos_1, height)
    site_2_photons = create_electroluminescence_photons(n_2, wavelength, pos_2, height)

    return site_1_photons + site_2_photons


def plot_photons(photons: Photons):
    """Plot the photons with direction and polarization vectors"""
    import matplotlib.pyplot as plt
    plt.style.use('~/styles/clarke-default.mplstyle')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = plt.style.library['fivethirtyeight']['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(photons.pos[:, 0], photons.pos[:, 1], photons.pos[:, 2], c=colors[0], label='Position')

    ax.quiver(photons.pos[:, 0], photons.pos[:, 1], photons.pos[:, 2],
              photons.dir[:, 0], photons.dir[:, 1], photons.dir[:, 2],
              color=colors[1], alpha=1., length=3., lw=1, normalize=True, label='Direction')

    ax.quiver(photons.pos[:, 0], photons.pos[:, 1], photons.pos[:, 2],
              photons.pol[:, 0], photons.pol[:, 1], photons.pol[:, 2],
              color=colors[2], alpha=1., length=3., lw=1, normalize=True, label='Polarization')

    ax.set_xlabel('x [mm]', labelpad=15)
    ax.set_ylabel('y [mm]', labelpad=15)
    ax.set_zlabel('z [mm]', labelpad=20)
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([-389, -369])
    ax.set_box_aspect((1, 1, 1))
    ax.dist = 12
    ax.legend()
    ax.set_title('Electroluminescence Photons')
    return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sites = np.load('chroma-lxe/data/XeNu_LXe_surface_points_site2.npy')
    #print(sites[0])
    # photons = create_multisite_electroluminescence_photons(50, 175, sites[1], sites[2], 10.0)
    photons = create_electroluminescence_photons(50, 175, sites[5], 10.0)
    fig, ax = plot_photons(photons)
    #plt.savefig('ms_event_photons.png')
    #plt.savefig('ss_event_photons.png')
    plt.show()
    
