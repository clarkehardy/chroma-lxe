import chroma
import numpy as np
from chroma.event import *
import logging
logging.getLogger('chroma').setLevel(logging.DEBUG)

if __name__ == '__main__':
    from chroma.sim import Simulation
    from chroma.sample import uniform_sphere
    from chroma.event import Photons
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import matplotlib.pyplot as plt

    import sys
    sys.path.append('../geometry')

    from geometry import build_detector_from_yaml
    
    photon_pos = (0.,0.,0.)
    
    g = build_detector_from_yaml('../geometry/stl/detector.yaml', flat=True)
    g.bvh = load_bvh(g)

    sim = Simulation(g,geant4_processes=0, seed=123)

    # photon bomb from center
    def photon_bomb(n,wavelength,pos):
        pos = np.tile(pos,(n,1))
        dir = uniform_sphere(n)
        pol = np.cross(dir,uniform_sphere(n))
        wavelengths = np.repeat(wavelength,n)
        return Photons(pos,dir,pol,wavelengths)

    # vtx = Vertex('e-', [20,300,-30], [0,1,0], 100)
    def count_test(flags, test, none_of=None):
        if none_of is not None:
            has_test = np.bitwise_and(flags, test) == test
            has_none_of = np.bitwise_and(flags, none_of) == 0
            return np.count_nonzero(np.logical_and(has_test, has_none_of))
        else:
            return np.count_nonzero(np.bitwise_and(flags, test) == test)

    # writer = RootWriter('out.root', g)
    for ev in sim.simulate([photon_bomb(10000, 500, photon_pos)],keep_photons_beg=False,keep_photons_end=True,run_daq=False,max_steps=100):
        # write the python event to a root file
        # f.write_event(ev)
        detected = (ev.photons_end.flags & SURFACE_DETECT).astype(bool)

        photon_detection_efficiency = detected.sum()/len(detected)
        print('in event loop')
        print('# detected', detected.sum(), '# photons', len(detected))
        print(ev.photons_end.pos)
        print("fraction of detected photons: %f"%photon_detection_efficiency)

        p_flags = ev.photons_end.flags
        print("\tDetect", count_test(p_flags, SURFACE_DETECT))
        print("\tScintillation", count_test(p_flags, SCINTILLATION))
        print("\tScintillation&SurfaceDetect", count_test(p_flags, SCINTILLATION | SURFACE_DETECT))
        print("\tCherenkov", count_test(p_flags, CHERENKOV))
        print("\tCherenkov No Reemit", count_test(p_flags, CHERENKOV, none_of=BULK_REEMIT))
        print("\tCherenkov&SurfaceAbsorb", count_test(p_flags, CHERENKOV | SURFACE_ABSORB))
        print("\tCherenkov&SurfaceDetect", count_test(p_flags, CHERENKOV | SURFACE_DETECT))
        print("\tCherenkov&SurfaceDetect No Reemit", count_test(p_flags, CHERENKOV | SURFACE_DETECT, none_of=BULK_REEMIT))
        print("\tCherenkov&BulkReemit&SurfaceDetect", count_test(p_flags, CHERENKOV | BULK_REEMIT | SURFACE_DETECT))
        print("\tCherenkov&Rayleigh&SurfaceDetect", count_test(p_flags, CHERENKOV | RAYLEIGH_SCATTER | SURFACE_DETECT))
        print(
            "\tCherenkov&BulkReemit&Rayleigh&SurfaceDetect",
            count_test(p_flags, CHERENKOV | BULK_REEMIT | RAYLEIGH_SCATTER | SURFACE_DETECT),
        )
        print("\tCherenkov&BulkAbsorb", count_test(p_flags, CHERENKOV | BULK_ABSORB))
        print("\tCherenkov&BulkReemit", count_test(p_flags, CHERENKOV | BULK_REEMIT))
        print("\tCherenkov&Rayleigh", count_test(p_flags, CHERENKOV | RAYLEIGH_SCATTER))
        print("\tCherenkov&BulkReemit&Rayleigh", count_test(p_flags, CHERENKOV | BULK_REEMIT | RAYLEIGH_SCATTER))
        print("\tNoHit", count_test(p_flags, NO_HIT))
        print("\tAbort", count_test(p_flags, NAN_ABORT))
        print('\tSurfaceAbsorb', count_test(p_flags, SURFACE_ABSORB))
        print('\tBulkAbsorb', count_test(p_flags, BULK_ABSORB))