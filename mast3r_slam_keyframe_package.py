from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class KeyframePacket:
    frame_id: int
    image: np.ndarray  # H x W x 3 (uint8)
    points_world: np.ndarray
    conf: np.ndarray  # H x W or per-point
    K: np.ndarray
    
  