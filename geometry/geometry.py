import glob
import yaml
import numpy as np
import hashlib

import chroma.make as make
from chroma import geometry
from chroma.detector import Detector
from chroma.transform import make_rotation_matrix
from chroma.loader import mesh_from_stl
from chroma.loader import create_geometry_from_obj
from chroma.cache import Cache
from chroma.camera import view, EventViewer
import materials
import surfaces
import utils

import logging

log = logging.getLogger(__file__.split('/')[-1].split('.')[0])
log.setLevel(logging.INFO)


class BBox:
    def __init__(self, vertices_or_min=[np.inf,np.inf,np.inf], max=[-np.inf,-np.inf,-np.inf]):
        if np.shape(vertices_or_min) == (3,):
            self.min = np.asarray(vertices_or_min)
            self.max = np.asarray(max)
        else:
            self.min = np.min(vertices_or_min, axis=0)
            self.max = np.max(vertices_or_min, axis=0)

        assert len(self.min) == len(self.max) == 3

    def __repr__(self):
        return f'BBox(min={self.min}, max={self.max})'
    
    def __add__(self, other):
        new_min = np.c_[self.min, other.min].min(axis=1)
        new_max = np.c_[self.max, other.max].max(axis=1)
        return BBox(new_min, new_max)
    
    def as_mesh(self):
        box_center = (self.min + self.max) / 2
        dx,dy,dz = self.extent

        return make.box(dx, dy, dz, center=box_center)
    
    @property
    def extent(self):
        return self.max - self.min
    
def add_cavity(g, bbox, material1=materials.lxe, material2=materials.vacuum, surface=surfaces.reflect0):
    cavity = make.cylinder_along_z(bbox.extent.max(), 2*bbox.extent.max())
    rot_normal = utils.gen_rot(
        [0, 0, 1],
        [
            1 if i == bbox.extent.argmax() else 0 for i in range(3)
        ],  # i.e. (1,0,0), (0,1,0), or (0,0,1)
    )

    solid = geometry.Solid(cavity, material1, material2, surface=surface, color=0xF0CCCCCC)
    g.add_solid(solid, rotation=rot_normal, displacement=bbox.min + bbox.extent/2)

def build_detector_from_yaml(config_path, flat=True, save_cache=True, load_cache=True):
    if load_cache:
        g = cache_load(config_path)
        if g: return create_geometry_from_obj(g, auto_build_bvh=False)

    config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)

    target = config.get('target',  'vacuum')
    target = getattr(materials, target)

    g = Detector(target)
    
    solid_bbox = BBox()
    for i,part in enumerate(config['parts']):
        opts = part['options']

        curr = i+1
        total = len(config['parts'])
        print(f"[{curr}/{total}] building part {part['name']}")
        
        path = sorted(list(glob.glob(opts['path'])))

        if not opts['rotation']['angle']:
            rotation = np.eye(3)
        else:
            rotation = make_rotation_matrix(
                opts['rotation']['angle']*np.pi/180.0,
                opts['rotation']['dir']
            )

        translation = opts['translation']

        # load materials
        material_kwargs = {}
        for keyword in ('material1', 'material2', 'surface'):
            lib = materials if 'material' in keyword else surfaces
            material_kwargs[keyword] = getattr(lib, opts['material'][keyword])
        material_kwargs['color'] = opts['material'].get('color', 0xFFFFFF)

        for p in path:
            if config['log']:
                log.info('  loading %s', p.split('/')[-1])
            
            mesh = mesh_from_stl(p)
            solid = geometry.Solid(mesh, **material_kwargs)
            solid_bbox += BBox(mesh.vertices)

            if opts['is_detector']:
                g.add_pmt(solid, rotation, translation)
            else:
                g.add_solid(solid, rotation, translation)

    add_cavity(g, solid_bbox)

    if flat:
        g = create_geometry_from_obj(g)

    if save_cache:
        cache_save(g, config_path)

    return g

def cache_load(path):
    c = Cache()
    md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
    if md5 in c.list_geometry():
        log.info('loading geometry from cache')
        return c.load_geometry(md5)
    else:
        log.info('geometry not in cache. building from yaml...')
        return None
    
def cache_save(g, path):
    log.info('saving geometry to cache')
    c = Cache()
    md5 = hashlib.md5(open(path, 'rb').read()).hexdigest()
    c.save_geometry(md5, g)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build and view a detector from a yaml file')
    parser.add_argument('yaml', type=str, help='path to the yaml file')
    parser.add_argument('--no-cache', action='store_true', help='do not load from cache')
    # optional event file input
    parser.add_argument('-i', '--input', type=str, help='path to the event file (ROOT) to visualize')

    args = parser.parse_args()

    g = build_detector_from_yaml(args.yaml, load_cache=not args.no_cache)

    if args.input is None:
        from chroma.camera import Camera
        cam = Camera(g)
        cam.run()
    else:
        import pygame  # sy (5/21/24) this avoids a segfault on turning on the camera. I don't know why. Don't ask.
        pygame.init()

        viewer = EventViewer(g, args.input, size=(1500, 1500),
                             background=0x000000)
        viewer.run()