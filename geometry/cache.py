import hashlib
import logging
import os
from pathlib import Path

from chroma.cache import Cache as ChromaCache
from chroma.detector import Detector

__all__ = ["GeometryCache"]

log = logging.getLogger(__name__)

class GeometryCache:
    """A cache for detector geometries.
    
    This utilizes chroma's cache system to store detector geometries. This thin wrapper
    loads and saves detector geometries to the cache based on chroma-lxe based
    detector specifications.
    """
    
    def __init__(self, cache_path: Path = Path("~/.chroma/").expanduser()):
        """Initializes the cache.
        
        Parameters
        ----------
        cache_path : Path
            The path to the cache directory. Default is ~/.chroma/.
        """
        
        self.chroma_cache = ChromaCache(cache_path)

    def load(self, config_path: Path) -> Detector:
        md5 = self._calculate_md5(config_path)
        if md5 in self.chroma_cache.list_geometry():
            log.info("Loading geometry from cache")
            return self.chroma_cache.load_geometry(md5)
        else:
            log.info("Geometry not in cache")
            return None

    def save(self, detector: Detector, config_path: Path):
        log.info("Saving geometry to cache")
        md5 = self._calculate_md5(config_path)
        self.chroma_cache.save_geometry(md5, detector)

    @staticmethod
    def _calculate_md5(path: Path) -> str:
        if isinstance(path, str):
            path = Path(path)
        
        return hashlib.md5(path.read_bytes()).hexdigest()
