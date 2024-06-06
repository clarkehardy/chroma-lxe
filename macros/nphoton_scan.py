from ast import arg
from turtle import pos
import chroma
import numpy as np
from chroma.event import *
from chroma.sample import uniform_sphere
from chroma.event import Photons
import os 
import time
from chroma.sim import Simulation
from chroma.loader import load_bvh
from chroma.generator import vertex
import h5py

import sys
sys.path.append('../geometry')
from geometry import build_detector_from_yaml

import logging
logging.getLogger('chroma').setLevel(logging.DEBUG)

class CSVLogger:
    def __init__(self, filename, variables):
        if not os.path.exists(filename):
            self.f = open(filename, 'w')
            self.f.write(','.join(variables) + '\n')
        else:
            self.f = open(filename, 'a')
        self.variables = variables

    def write(self, **kwargs):
        self.f.write(','.join([f"{kwargs[v]:.8f}" for v in self.variables]) + '\n')

    def close(self):
        self.f.close()    

class H5Logger:
    def __init__(self, filename, variables):
        self.filename = filename
        self.variables = variables

        if os.path.exists(filename):
            logging.warning(f"File {filename} already exists. Overwriting.")
            os.remove(filename)
        self.f = h5py.File(filename, 'a')

        for var in variables:
            if var not in self.f:
                self.f.create_dataset(var, (0,), maxshape=(None,), dtype='f')

    def write(self, **kwargs):
        for var in self.variables:
            data = self.f[var]
            data.resize((data.shape[0] + 1,))
            data[-1] = kwargs[var]

    def close(self):
        self.f.close()

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run a test simulation')
    parser.add_argument('--path', default='../geometry/config/detector.yaml', help='Detector yaml file')
    parser.add_argument('-o', '--output', default='out.root', help='Where to save PTE data.')
    parser.add_argument('-n', type=int, default=100, help='Number of photons to simulate')
    # argument for whether to use a single channel or individual channels
    parser.add_argument('--single_channel', action='store_true', help='Treat all detecting channels as one')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()
    return args

# photon bomb from center
def photon_bomb(n,wavelength,pos):
    pos = np.tile(pos,(n,1))
    dir = uniform_sphere(n)
    pol = np.cross(dir,uniform_sphere(n))
    wavelengths = np.repeat(wavelength,n)
    return Photons(pos,dir,pol,wavelengths)

def main():
    from tqdm import tqdm
    import chroma
    from chroma.io.root import RootWriter

    t0 = time.time()
    args = parse_args()
    
    photon_pos = np.load('../data/lightmap_points_2mm.npy')
    
    g = build_detector_from_yaml(args.path, flat=True)
    g.bvh = load_bvh(g, read_bvh_cache=True)

    sim = Simulation(g,geant4_processes=0, seed=123, photon_tracking=False)
    simulated = 0

    n_channels = g.num_channels()

    def zpad(ch):
        return str(ch).zfill(2)

    variables = ['posX', 'posY', 'posZ', 'n', 'detected', 'pte']
    if not args.single_channel:
        for i in range(n_channels):
            variables += [f'ch{zpad(i)}_detected', f'ch{zpad(i)}_pte']
    variables += ['time_spent']

    logger = H5Logger('../data/nphoton_scan_2mm.h5', variables)

    # rw = RootWriter(args.output)
    t0 = time.time()
    bomb = photon_bomb(args.n, 175, photon_pos[0])
    for i in tqdm(range(len(photon_pos)), desc='Scanning', total=len(photon_pos)):
        bomb.pos[:] = photon_pos[i]
        t0 = time.time()
        evs = list(sim.simulate(bomb, keep_photons_beg=False,keep_photons_end=False,max_steps=100,
                           keep_hits=True,
                           run_daq=False,
                           photons_per_batch=50000))
        for ev in evs:
            out = {}
            pos = photon_pos[i]
            out['posX'] = pos[0]
            out['posY'] = pos[1]
            out['posZ'] = pos[2]
            out['n'] = args.n

            detected = len(ev.flat_hits)
            out['detected'] = detected
            out['pte'] = detected/args.n

            if not args.single_channel:
                for c in range(n_channels):
                    hits = ev.hits
                    detected = len(hits.get(c, []))
                    out[f'ch{zpad(c)}_detected'] = detected
                    out[f'ch{zpad(c)}_pte'] = detected/args.n
            t = time.time()
            out['time_spent'] = t-t0
            logger.write(**out)

            if i == 10:
                logger.close()
                sys.exit(0)
        t0 = t
        # rw.write_event(ev)
    # rw.close()
    logger.close()


if __name__ == "__main__":
    main()
