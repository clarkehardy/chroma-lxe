import glob
from chroma import geometry
from chroma.detector import Detector
from chroma.transform import make_rotation_matrix
from chroma.loader import mesh_from_stl
import chroma.make as make
import yaml
import numpy as np

from chroma.camera import view
from chroma.loader import create_geometry_from_obj
import materials
import surfaces

import logging
log = logging.getLogger(__file__.split('/')[-1].split('.')[0])
log.setLevel(logging.INFO)

def build_detector_from_yaml(path, flat=True):

    config = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)

    target = config.get('target',  'vacuum')
    target = getattr(materials, target)

    g = Detector(target)

    
    mesh = make.cylinder_along_z(1000, 2*1000)
    cavity = geometry.Solid(mesh, target, materials.vacuum, surface=surfaces.reflect0, color=0xFF000000)
    
    g.add_solid(cavity)

    channel_id = 0
    for part in config['parts']:
        opts = part['options']
        if config['log']:
            log.info('building part %s', part['name'])
        
        path = list(glob.glob(opts['path']))

        if opts['rotation']['angle'] == 0:
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
        material_kwargs['color'] = opts.get('color', 0xFFFFFF)

        for p in path:
            if config['log']:
                log.info('  loading %s', p)
            
            mesh = mesh_from_stl(p)
            
            solid = geometry.Solid(mesh, **material_kwargs)

            if is_det:
                g.add_pmt(solid, rotation, translation, channel_id)
            else:
                g.add_solid(solid, rotation, translation)

    if flat:
        g = create_geometry_from_obj(g)

    return g


if __name__ == '__main__':
    import sys
    from chroma.camera import Camera

    g = build_detector_from_yaml(sys.argv[1])

    camera = Camera(g, size=(1000,1000))
    camera.start()
    camera.run()