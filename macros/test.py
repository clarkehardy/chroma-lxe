from ast import arg
import chroma
import numpy as np
from chroma.event import *
from chroma.sample import uniform_sphere
from chroma.event import Photons

import logging
logging.getLogger('chroma').setLevel(logging.DEBUG)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run a test simulation')
    # detector yaml. default is the one in the geometry directory
    parser.add_argument('--path', default='../geometry/config/detector.yaml', help='Detector yaml file')
    parser.add_argument('-o', '--output', default=None, help='Save events to a root file')
    # -n or --nphotons
    parser.add_argument('-n', type=int, default=100, help='Number of photons to simulate')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')


    args = parser.parse_args()
    return args

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

def print_stats(ev):
    detected = (ev.photons_end.flags & SURFACE_DETECT).astype(bool)
    photon_detection_efficiency = detected.sum()/len(detected)
    print('in event loop')
    print('# detected', detected.sum(), '# photons', len(detected))
    print("fraction of detected photons: %f"%photon_detection_efficiency)

    p_flags = ev.photons_end.flags
    print("\tDetect", count_test(p_flags, SURFACE_DETECT))
    print("\tNoHit", count_test(p_flags, NO_HIT))
    print("\tAbort", count_test(p_flags, NAN_ABORT))
    print('\tSurfaceAbsorb', count_test(p_flags, SURFACE_ABSORB))
    print('\tBulkAbsorb', count_test(p_flags, BULK_ABSORB))




def main():
    args = parse_args()

    from chroma.sim import Simulation
    from chroma.loader import load_bvh
    from chroma.generator import vertex

    import sys
    sys.path.append('../geometry')

    from geometry import build_detector_from_yaml
    
    photon_pos = (0.,-10.,0.)
    
    g = build_detector_from_yaml(args.path, flat=True)
    g.bvh = load_bvh(g, read_bvh_cache=True)

    sim = Simulation(g,geant4_processes=0, seed=args.seed, photon_tracking=True)

    if args.output:
        from chroma.io.root import RootWriter
        f = RootWriter(args.output, g)
    for ev in sim.simulate([photon_bomb(args.n, 175, photon_pos)],keep_photons_beg=False,keep_photons_end=True,run_daq=False,max_steps=100,
                           keep_hits=False):
        if args.output:
            f.write_event(ev)
        print_stats(ev)
        # pte = len(ev.flat_hits)/args.n
        # print(pte)

    if args.output:
        f.close()



if __name__ == "__main__":
    # Cprofile

    import cProfile
    cProfile.run('main()', 'profile.out')

