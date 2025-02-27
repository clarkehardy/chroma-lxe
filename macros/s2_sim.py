from timeit import default_timer as timer

import numpy as np
from chroma.event import *
from generator.photons import create_photon_bomb, create_electroluminescence_photons, create_multisite_electroluminescence_photons
from geometry.builder import build_detector_from_yaml
from utils.output import H5Logger, print_table
import time


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

    db.positions_path = "/home/clarke/chroma-lxe/data/XeNu_LXe_surface_points.npy"
    db.positions_path_2 = "/home/clarke/chroma-lxe/data/XeNu_LXe_surface_points_site2.npy"
    db.extraction_height = 6.5 # mm
    db.wavelength = 175
    db.single_site = False
    pmts = False
    db.sensors = 'PMTs' if pmts else 'SiPMs'
    db.output_file = "s2_sim_test_" + db.sensors + ["_m","_s"][int(db.single_site)] + "s.h5"

    db.config_file = "/home/clarke/chroma-lxe/geometry/config/XeNu_" + db.sensors + ".yaml"
    db.num_events = 10_000
    db.n_photons = 50_000
    db.notify_event = 10
    db.single_channel = False


def __define_geometry__(db):
    """Returns a chroma Detector or Geometry"""
    geometry = build_detector_from_yaml(db.config_file, flat=True)
    db.geometry = geometry
    return geometry

def __event_generator__(db):
    """A generator to yield chroma Events (or something a chroma Simulation can
    convert to a chroma Event)."""
    if db.single_site:
        yield from (
            create_electroluminescence_photons(n=db.n_photons, wavelength=db.wavelength, pos=position, \
                                               height=db.extraction_height) for position in db.photon_positions
        )
    else:
        yield from (
            create_multisite_electroluminescence_photons(n=db.n_photons, wavelength=db.wavelength, pos_1=position, \
                                                         pos_2=position_2, height=db.extraction_height) for position, position_2 \
                                                         in zip(db.photon_positions, db.photon_positions_2)
        )


def __simulation_start__(db):
    """Called at the start of the event loop"""
    db.ev_idx = 0
    db.t_sim_start = timer()

    db.photon_positions = np.load(db.positions_path)[:db.num_events]
    db.num_events = len(db.photon_positions)
    db.n_channels = db.geometry.num_channels()
    db.photon_positions_2 = np.load(db.positions_path_2)[:db.num_events]
    
    # create variable labels
    variables = ["posX", "posY", "posZ", "n", "detected", "pte"]
    if not db.single_site:
        variables += ["posX_2", "posY_2", "posZ_2"]
    if not db.single_channel:
        zfill_width = int(np.log10(db.n_channels)) + 1
        for i in range(db.n_channels):
            channel_id = str(i).zfill(zfill_width)
            variables += [f"ch{channel_id}_detected", f"ch{channel_id}_pte"]
    variables += ["time_spent"]
    db.writer = H5Logger(db.output_file, variables)

    db.event_idx = 0
    db.total_detected = 0
    db.total_pte = 0
    db.total_time = 0
    db.start_time = time.time()


def __process_event__(db, ev):
    """Called for each generated event"""
    output = {}
    position = db.photon_positions[db.event_idx]
    output["posX"] = position[0]
    output["posY"] = position[1]
    output["posZ"] = position[2]
    output["n"] = db.n_photons

    if not db.single_site:
        position2 = db.photon_positions_2[db.event_idx]
        output["posX_2"] = position2[0]
        output["posY_2"] = position2[1]
        output["posZ_2"] = position2[2]

    detected = len(ev.flat_hits)
    output["detected"] = detected
    output["pte"] = detected / db.n_photons

    if not db.single_channel:
        zfill_width = int(np.log10(db.n_channels)) + 1
        for c in range(db.n_channels):
            channel_id = str(c).zfill(zfill_width)
            hits = ev.hits
            channel_detected = len(hits.get(c, []))
            output[f"ch{channel_id}_detected"] = channel_detected
            output[f"ch{channel_id}_pte"] = channel_detected / db.n_photons

    ev_time = time.time() - db.start_time
    output["time_spent"] = ev_time
    db.writer.write(**output)

    db.total_detected += output["detected"]
    db.total_pte += output["pte"]
    db.start_time += ev_time
    db.total_time += ev_time
    db.event_idx += 1


def __simulation_end__(db):
    """Called at the end of the event loop"""
    db.writer.close()

    n_positions = len(db.photon_positions)
    results = dict(
        output_path=db.output_file,
        n_positions=n_positions,
        n_photons_per_position=db.n_photons,
        n_detected=db.total_detected,
        n_detected_per_position=db.total_detected / n_positions,
        avg_pte_per_position=db.total_pte / n_positions,
        total_time=db.total_time,
        sec_per_position=db.total_time / n_positions,
        positions_per_sec=n_positions / db.total_time,
        photons_per_sec=n_positions * db.n_photons / db.total_time,
    )
    print_table(**results)