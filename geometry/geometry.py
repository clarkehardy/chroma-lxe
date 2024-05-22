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

import logging


log = logging.getLogger(__file__.split('/')[-1].split('.')[0])
log.setLevel(logging.INFO)

def build_detector_from_yaml(config_path, flat=True, save_cache=True, load_cache=True):
    if load_cache:
        g = cache_load(config_path)
        if g: return create_geometry_from_obj(g, auto_build_bvh=False)

    config = yaml.load(open(config_path, 'r'), Loader=yaml.SafeLoader)

    target = config.get('target',  'vacuum')
    target = getattr(materials, target)

    g = Detector(target)
    
    mesh = make.cylinder_along_z(1000, 2*1000)
    cavity = geometry.Solid(mesh, target, materials.vacuum, surface=surfaces.reflect0, color=0xC8CCCCCC)
    
    g.add_solid(cavity)

    channel_id = 0
    for i,part in enumerate(config['parts']):
        opts = part['options']

        curr = i+1
        total = len(config['parts'])
        print(f"[{curr}/{total}] building part {part['name']}")
        
        path = list(glob.glob(opts['path']))

        if not opts['rotation']['angle']:
            rotation = np.eye(3)
        else:
            rotation = make_rotation_matrix(
                opts['rotation']['angle']*np.pi/180.0,
                opts['rotation']['dir']
            )

        translation = opts['translation']

        is_det = opts['is_detector']

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

            if is_det:
                g.add_pmt(solid, rotation, translation)
            else:
                g.add_solid(solid, rotation, translation)

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

        viewer = EventViewer(g, args.input, size=(1200, 1200))
        viewer.run()