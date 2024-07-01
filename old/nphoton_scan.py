from __future__ import annotations

import argparse
import math
import sys
import time
from typing import List, Tuple

import chroma
import numpy as np
from chroma.event import Photons
from chroma.loader import load_bvh
from chroma.sample import uniform_sphere
from chroma.sim import Simulation
from tqdm import tqdm

sys.path.append('../geometry')
sys.path.append('../utils')
from builder import build_detector_from_yaml
from log import logger
from output import H5Logger, print_table


class LightmapSimulation:
    def __init__(self, config_file: str, seed: int = None, dry: bool = False):
        self.config_file = config_file
        self.seed = seed
        self.simulation, self.n_channels = self._setup_simulation(dry)
        self.variables = None  # set on-the-fly in run()

    def _setup_simulation(self, dry: bool = False) -> Tuple[Simulation, int]:
        """If a dry run, we'll save some extra information for visualization."""
        
        geometry = build_detector_from_yaml(self.config_file, flat=True)
        geometry.bvh = load_bvh(geometry, read_bvh_cache=True)
        self.geometry = geometry if dry else None
        simulation = Simulation(geometry, geant4_processes=0, seed=self.seed, photon_tracking=dry)
        n_channels = geometry.num_channels()
        return simulation, n_channels

    @staticmethod
    def create_photon_bomb(n: int, wavelength: float, pos: np.ndarray) -> Photons:
        pos = np.tile(pos,(n,1))
        dir = uniform_sphere(n)
        pol = np.cross(dir,uniform_sphere(n))
        wavelengths = np.repeat(wavelength,n)
        return Photons(pos,dir,pol,wavelengths)

    def _simulate(self, photons: Photons, single_channel: bool, dry: bool = False) -> List:
        return list(
            self.simulation.simulate(
                photons,
                max_steps=100,
                keep_hits=not single_channel,
                photons_per_batch=1_000_000,
                keep_photons_beg=dry,
                keep_photons_end=dry,
                run_daq=dry,
            )
        )

    def _process_events(
        self, events: List, n_photons: int, position: np.ndarray, single_channel: bool
    ) -> dict:
        output = {k: None for k in self.variables}
        output['posX'] = position[0]
        output['posY'] = position[1]
        output['posZ'] = position[2]
        output['n'] = n_photons

        for event in events:
            detected = len(event.flat_hits)
            output["detected"] = detected
            output["pte"] = detected / n_photons

            if not single_channel:
                zfill_width = int(math.log10(self.n_channels)) + 1
                for c in range(self.n_channels):
                    channel_id = str(c).zfill(zfill_width)
                    hits = event.hits
                    channel_detected = len(hits.get(c, []))
                    output[f"ch{channel_id}_detected"] = channel_detected
                    output[f"ch{channel_id}_pte"] = channel_detected / n_photons

        return output

    @staticmethod
    def _get_output_variables(n_channels, single_channel: bool) -> List[str]:
        variables = ['posX', 'posY', 'posZ', 'n', 'detected', 'pte']
        if not single_channel:
            zfill_width = int(math.log10(n_channels)) + 1
            for i in range(n_channels):
                channel_id = str(i).zfill(zfill_width)
                variables += [f"ch{channel_id}_detected", f"ch{channel_id}_pte"]
        variables += ['time_spent']
        return variables

    def run(
        self,
        photon_positions: np.ndarray,
        output_file: str,
        n_photons: int,
        single_channel: bool = False,
    ):
        self.variables = self._get_output_variables(self.n_channels, single_channel)
        writer = H5Logger(output_file, self.variables)

        total_detected = 0
        total_pte = 0
        total_time = 0

        start_time = time.time()
        for i in tqdm(range(len(photon_positions)), desc="Scanning"):
            bomb_start_time = time.time()
            bomb = self.create_photon_bomb(n_photons, 175, photon_positions[i])
            events = self._simulate(bomb, single_channel)
            output = self._process_events(
                events, n_photons, photon_positions[i], single_channel
            )
            bomb_time = time.time() - bomb_start_time
            output["time_spent"] = bomb_time
            writer.write(**output)

            total_detected += output["detected"]
            total_pte += output["pte"]
            total_time += bomb_time

            if i == 100 and total_detected == 0:
                logger.warning(
                    "\nNo photons detected after 100 positions. Are there any detectors?"
                    " Ensure that is_detector is set to True in the detector configuration"
                    " and that the detector's surface contains the detect property with a "
                    "value > 0.0."
                )


        writer.close()
        
        return dict(
            output_path=output_file,
            n_positions=len(photon_positions),
            n_photons_per_position=n_photons,
            n_detected=total_detected,
            n_detected_per_position=total_detected / len(photon_positions),
            avg_pte_per_position=total_pte / len(photon_positions),
            total_time=total_time,
            sec_per_position=total_time / len(photon_positions),
            positions_per_sec=len(photon_positions) / total_time,
            photons_per_sec=len(photon_positions) * n_photons / total_time,
        )
    
    def visualize_bomb(self,
                       photon_position: np.ndarray, 
                       n_photons: int, 
                       single_channel: bool = False):
        from chroma.camera import EventViewer

        bomb: Photons = self.create_photon_bomb(n=n_photons,
                                                wavelength=175,
                                                pos=photon_position)
        event: List = self._simulate(bomb, single_channel, dry=True)
        # viewer = EventViewer(self.geometry, event, size=(1000,1000))
        # viewer.run()
        return event

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a photon simulation")
    parser.add_argument( "-C", "--config", default="../geometry/config/detector.yaml", help="Detector yaml file", )
    parser.add_argument( "-P", "--positions", required=True, help="Photon positions file" )
    parser.add_argument( "-O", "--output", help="Output file for PTE data" )
    parser.add_argument( "-N", type=int, default=100, help="Number of photons to simulate" )
    parser.add_argument( "--single-channel", action="store_true", help="Treat all detecting channels as one", )
    parser.add_argument( "--dry", action="store_true", help="Dry run. Simulate a bomb at the first position and visualize it.")
    parser.add_argument( "--seed", type=int, default=None, help="Random seed for simulation" )
    args = parser.parse_args()
    if not args.output and not args.dry:
        parser.error("Output file is required unless --dry is set.")
    return args

def main():
    """Run a photon simulation."""
    args = parse_arguments()

    photon_positions = np.load(args.positions)[:100]
    simulation = LightmapSimulation(args.config, args.seed, args.dry)
    
    if args.dry:
        ev = simulation.visualize_bomb(photon_positions[0], args.N)
        print(ev[0].photons_beg.pos[0])
        return
    
    output = simulation.run(photon_positions, args.output, args.N, args.single_channel)
    print_table(**output)

if __name__ == "__main__":
    main()
