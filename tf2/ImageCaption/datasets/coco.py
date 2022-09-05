from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import random
import xml.etree.ElementTree as ET
from typing import List
from typing import Tuple

class Dataset:
  def __init__(self, split, dir = "dataset/COCO/val2014", augment = True, shuffle = True, cache = True):
    if not os.path.exists(dir):
      raise FileNotFoundError("Dataset directory does not exist: %s" % dir)

    self.split      = split
    self._dir       = dir
    self._augment   = augment
    self._shuffle   = shuffle
    self._cache     = cache


