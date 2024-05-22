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
        self.f.write(','.join([str(kwargs[v]) for v in self.variables]) + '\n')

    def close(self):
        self.f.close()    

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run a test simulation')
    # detector yaml. default is the one in the geometry directory
    parser.add_argument('--path', default='../geometry/stl/detector.yaml', help='Detector yaml file')
    parser.add_argument('-o', '--output', default='out.csv', help='Where to save PTE data.')
    # -n or --nphotons
    parser.add_argument('-n', type=int, default=100, help='Number of photons to simulate')
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

    t0 = time.time()
    args = parse_args()
    
    photon_pos = (0.,-10.,0.)
    
    g = build_detector_from_yaml(args.path, flat=True)
    g.bvh = load_bvh(g, read_bvh_cache=True)

    sim = Simulation(g,geant4_processes=0, seed=None, photon_tracking=True)

    detected = 0
    simulated = 0

    logger = CSVLogger(args.output, ['posX', 'posY', 'posZ', 'n', 'detected', 'pte', 'time_spent'])

    for ev in sim.simulate([photon_bomb(args.n, 175, photon_pos)],keep_photons_beg=False,keep_photons_end=True,run_daq=False,max_steps=100,
                           keep_hits=False,
                           photons_per_batch=100000):
        detected += len(ev.flat_hits)
        simulated += args.n
    t1 = time.time()
    logger.write(posX=photon_pos[0],
                 posY=photon_pos[1],
                 posZ=photon_pos[2],
                 n=simulated,
                 detected=detected,
                 pte=detected/simulated,
                 time_spent=t1-t0)

    logger.close()
    print(simulated, detected, detected/simulated, t1-t0)   # needed, will be captured by jobrunner.py

if __name__ == "__main__":
    main()
