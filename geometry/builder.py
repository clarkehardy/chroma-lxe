from __future__ import annotations
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chroma.make as make
import materials
import numpy as np
import surfaces
import utils
import yaml
from bbox import BBox
from cache import GeometryCache
from chroma import geometry
from chroma.camera import EventViewer
from chroma.detector import Detector
from chroma.loader import create_geometry_from_obj, mesh_from_stl
from chroma.transform import make_rotation_matrix

__all__ = ["build_detector_from_yaml", "build_detector_from_config"]

log = logging.getLogger(__file__.split("/")[-1].split(".")[0])
log.setLevel(logging.INFO)

@dataclass
class Rotation:
    angle: float
    dir: List[float]

@dataclass
class Material:
    material1: str
    material2: str
    surface: str
    color: int
    alpha: float = 1.0

@dataclass
class PartConfig:
    name: str
    path: str
    rotation: Rotation
    translation: List[float]
    scale: float
    material: Material
    is_detector: bool

@dataclass
class DetectorConfig:
    target: str
    parts: List[PartConfig]
    log: bool


def load_config_from_yaml(config_path: Path) -> DetectorConfig:
    """Loads a detector configuration from a yaml file and converts
    it to a DetectorConfig data class.
    
    The yaml definition follows a strict structure. Here is an example:
        
        ```yaml
        target: vacuum
        parts:
            - name: ...
              path: ...
              rotation: ...
              translation: ...
              scale: ...
              material: ...
              is_detector: ...
    """

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return DetectorConfig(
        target=config_dict.get("target", "vacuum"),
        parts=[PartConfig(**part) for part in config_dict["parts"]],
        log=config_dict.get("log", False),
    )


def build_detector_from_yaml(
    config_path: str | Path,
    flat: bool = True,
    load_cache: bool | str = True,
) -> Detector:
    """Builds a detector from a yaml file.

    Parameters
    ----------
    config_path : Path
        Path to the yaml file containing the detector definition.
    flat : bool
        If `True`, the detector will be flattened into a single mesh.
    load_cache : bool | str
        If `True`, it will attempt to load the detector from the cache to save time.
        If the detector is not in the cache, it will build the detector and save it
        to the cache. If a string is provided, it will be used as the cache path.
        The default cache path is `~/.chroma/cache`.
        
    Returns
    -------
    chroma.Detector
        The detector object.
    """
    if load_cache:
        cache = GeometryCache()
        cached_detector = cache.load(config_path)
        if cached_detector:
            return create_geometry_from_obj(cached_detector, auto_build_bvh=False)

    config = load_config_from_yaml(config_path)
    detector = build_detector_from_config(config, flat)

    if load_cache:
        cache.save(detector, config_path)

    return detector


def build_detector_from_config(config: DetectorConfig, flat: bool = True) -> Detector:
    """Builds a detector from a DetectorConfig object."""

    target_material = getattr(materials, config.target)
    detector = Detector(target_material)

    solid_bbox = build_detector_parts(detector, config)
    add_cavity_from_bbox(detector, solid_bbox)

    if flat:
        detector = create_geometry_from_obj(detector)

    return detector


def build_detector_parts(detector: Detector, config: DetectorConfig) -> BBox:
    """Builds and adds parts to the detector. Returns the bounding box of all parts."""

    solid_bbox = BBox()
    for i, part in enumerate(config.parts, 1):
        log.info(f"[{i}/{len(config.parts)}] building part {part.name}")

        rotation = (
            make_rotation_matrix(part.rotation["angle"] * np.pi / 180.0, part.rotation["dir"])
            if part.rotation["angle"]
            else np.eye(3)
        )

        material_kwargs = prepare_material_kwargs(part.material)

        for p in sorted(glob.glob(part.path)):
            if config.log:
                log.info(f"\tloading {p}")

            mesh = mesh_from_stl(p)
            solid = geometry.Solid(mesh, **material_kwargs)
            solid_bbox += BBox(mesh.vertices)

            if part.is_detector:
                detector.add_pmt(solid, rotation, part.translation)
            else:
                detector.add_solid(solid, rotation, part.translation)

    return solid_bbox


def prepare_material_kwargs(material_config: dict) -> dict:
    """Accesses materials and surfaces from our databases in materials.py and surfaces.py."""

    material_kwargs = {}
    for keyword in ("material1", "material2", "surface"):
        lib = materials if "material" in keyword else surfaces
        material_kwargs[keyword] = getattr(lib, material_config[keyword])
    material_kwargs["color"] = material_config.get("color", 0xFFFFFF)
    return material_kwargs


def add_cavity_from_bbox(
    g,
    bbox,
    material1=materials.lxe,
    material2=materials.vacuum,
    surface=surfaces.reflect0,
):
    """Creates a cylindrical envelope around the detector using the bounding box of the detector."""

    cavity = make.cylinder_along_z(bbox.extent.max(), 2 * bbox.extent.max())
    rot_normal = utils.gen_rot([0, 0, 1], np.eye(3)[bbox.extent.argmax()])
    solid = geometry.Solid(
        cavity, material1, material2, surface=surface, color=0xF0CCCCCC
    )
    g.add_solid(solid, rotation=rot_normal, displacement=bbox.min + bbox.extent / 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and view a detector from a yaml file"
    )
    parser.add_argument("yaml", type=str, help="path to the yaml file")
    parser.add_argument(
        "--no-cache", action="store_true", help="do not load from cache"
    )
    parser.add_argument(
        "-i", "--input", type=str, help="path to the event file (ROOT) to visualize"
    )

    args = parser.parse_args()

    g = build_detector_from_yaml(args.yaml, load_cache=not args.no_cache)

    if args.input is None:
        from chroma.camera import Camera

        cam = Camera(g)
        cam.run()
    else:
        import pygame  # sy (5/21/24) this avoids a segfault on turning on the camera. I don't know why. Don't ask.

        pygame.init()

        viewer = EventViewer(g, args.input, size=(1500, 1500), background=0x000000)
        viewer.run()
