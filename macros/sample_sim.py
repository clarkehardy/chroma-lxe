from timeit import default_timer as timer

import numpy as np
from chroma.event import *
from generator.photons import create_photon_bomb
from geometry.builder import build_detector_from_yaml


def test_mask(flags, test, none_of=None):
    if none_of is not None:
        has_test = np.bitwise_and(flags, test) == test
        has_none_of = np.bitwise_and(flags, none_of) == 0
        return np.logical_and(has_test, has_none_of)
    else:
        return np.bitwise_and(flags, test) == test


def count_test(flags, test, none_of=None):
    return np.count_nonzero(test_mask(flags, test, none_of=none_of))


def __configure__(db):
    """Modify fields in the database here"""

    db.chroma_g4_processes = 0
    db.chroma_keep_hits = True
    db.chroma_keep_flat_hits = True
    db.chroma_photon_tracking = False          # saves photons at each step of propagation
    db.chroma_particle_tracking = False        # saves particles at each step of propagation (e-, ...)
    db.chroma_photons_per_batch = 1_000_000
    db.chroma_max_steps = 100
    db.chroma_daq = True
    db.chroma_keep_photons_beg = False         # saves photons at the beginning of the event
    db.chroma_keep_photons_end = True          # saves photons at the end of the event

    db.config_file = "/home/sam/sw/chroma-lxe/geometry/config/detector.yaml"
    db.num_events = 100
    db.num_photons = 100000
    db.notify_event = 10
    db.event_pos = [0, 1, 0]


def __define_geometry__(db):
    """Returns a chroma Detector or Geometry"""
    geometry = build_detector_from_yaml(db.config_file, flat=True)
    db.geometry = geometry
    return geometry

def __event_generator__(db):
    """A generator to yield chroma Events (or something a chroma Simulation can
    convert to a chroma Event)."""
    yield from (
        create_photon_bomb(n=db.num_photons, wavelength=175, pos=db.event_pos) for _ in range(db.num_events)
    )


def __simulation_start__(db):
    """Called at the start of the event loop"""
    db.ev_idx = 0
    db.t_sim_start = timer()


def __process_event__(db, ev):
    """Called for each generated event"""
    db.ev_idx += 1
    if db.ev_idx % db.notify_event == 0:
        t_now = timer()
        print(db.ev_idx, (t_now - db.t_sim_start) / db.ev_idx)

    p_flags = ev.photons_end.flags
    print("\tDetect", count_test(p_flags, SURFACE_DETECT))
    print("\tNoHit", count_test(p_flags, NO_HIT))
    print("\tAbort", count_test(p_flags, NAN_ABORT))
    print("\tSurfaceAbsorb", count_test(p_flags, SURFACE_ABSORB))
    print("\tBulkAbsorb", count_test(p_flags, BULK_ABSORB))


def __simulation_end__(db):
    """Called at the end of the event loop"""
    db.t_sim_end = timer()
    print(
        "Ran %i events at %0.2f s/ev"
        % (db.ev_idx, (db.t_sim_end - db.t_sim_start) / db.ev_idx)
    )