]import chroma
import numpy as np
from chroma.event import *
from chroma.sample import uniform_sphere
from chroma.event import Photons

import logging

logging.getLogger("chroma").setLevel(logging.DEBUG)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run a test simulation")
    # detector yaml. default is the one in the geometry directory
    parser.add_argument(
        "-o", "--output", default=None, help="Save events to a root file"
    )
    # -n or --nphotons
    parser.add_argument(
        "-n", type=int, default=100, help="Number of photons to simulate"
    )
    # number of fibers
    parser.add_argument(
        "-f", "--nfibers", type=int, default=1, help="Number of fibers to simulate"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument('--no-cache', action='store_true', help='do not load from cache')

    args = parser.parse_args()
    return args


def count_test(flags, test, none_of=None):
    if none_of is not None:
        has_test = np.bitwise_and(flags, test) == test
        has_none_of = np.bitwise_and(flags, none_of) == 0
        return np.count_nonzero(np.logical_and(has_test, has_none_of))
    else:
        return np.count_nonzero(np.bitwise_and(flags, test) == test)


def print_stats(ev):
    detected = (ev.photons_end.flags & SURFACE_DETECT).astype(bool)
    photon_detection_efficiency = detected.sum() / len(detected)
    print("in event loop")
    print("# detected", detected.sum(), "# photons", len(detected))
    print("fraction of detected photons: %f" % photon_detection_efficiency)

    p_flags = ev.photons_end.flags
    print("\tDetect", count_test(p_flags, SURFACE_DETECT))
    print("\tNoHit", count_test(p_flags, NO_HIT))
    print("\tAbort", count_test(p_flags, NAN_ABORT))
    print("\tSurfaceAbsorb", count_test(p_flags, SURFACE_ABSORB))
    print("\tBulkAbsorb", count_test(p_flags, BULK_ABSORB))


def main():
    args = parse_args()

    from chroma.sim import Simulation
    from chroma.loader import load_bvh
    from chroma.generator import vertex
    import yaml
    import sys
    from tqdm import tqdm

    sys.path.append("../geometry")
    from fiber import M114L01
    from geometry import build_detector_from_yaml

    g = build_detector_from_yaml('../geometry/config/ea-hv_4_fibers_100mm_extended.yaml', flat=True, 
                                 load_cache=not args.no_cache)
    g.bvh = load_bvh(g, read_bvh_cache=True)

    sim = Simulation(g, geant4_processes=0, seed=args.seed, photon_tracking=True)

    posdir = yaml.safe_load(open('../data/stl/fibers/fiber_positions_100mm_extended.yaml', 'r'))
    fibers = []
    for i in range(4):
        fiber = M114L01(
            position=posdir[f'fiber_{i}']['position'],
            direction=posdir[f'fiber_{i}']['direction'],
        )
        fibers.append(fiber)

    if args.output:
        from chroma.io.root import RootWriter

        f = RootWriter(args.output, g)

    nbatches = args.n // 500000
    leftover = args.n % 500000
    N = [500000] * nbatches + ([leftover] if leftover else [])
    assert sum(N) == args.n

    for ev in sim.simulate(
        tqdm([fiber.generate_photons(n) for n in N for fiber in fibers]),
        keep_photons_beg=False,
        keep_photons_end=True,
        run_daq=True,
        max_steps=100,
        keep_hits=True,
    ):
        if args.output:
            f.write_event(ev)
        print_stats(ev)

    if args.output:
        f.close()


if __name__ == "__main__":
    # Cprofile


    # cProfile.run('main()', 'profile.out')
    main()
