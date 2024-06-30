import numpy as np
from chroma import make

__all__ = ["BBox"]

class BBox:
    def __init__(
        self,
        vertices_or_min=np.array([np.inf, np.inf, np.inf]),
        max=np.array([-np.inf, -np.inf, -np.inf]),
    ):
        if np.shape(vertices_or_min) == (3,):
            self.min = np.asarray(vertices_or_min)
            self.max = np.asarray(max)
        else:
            self.min = np.min(vertices_or_min, axis=0)
            self.max = np.max(vertices_or_min, axis=0)

        assert len(self.min) == len(self.max) == 3

    def __repr__(self):
        return f"BBox(min={self.min}, max={self.max})"

    def __add__(self, other):
        new_min = np.c_[self.min, other.min].min(axis=1)
        new_max = np.c_[self.max, other.max].max(axis=1)
        return BBox(new_min, new_max)

    def as_mesh(self):
        box_center = (self.min + self.max) / 2
        dx, dy, dz = self.extent

        return make.box(dx, dy, dz, center=box_center)

    @property
    def extent(self):
        return self.max - self.min
