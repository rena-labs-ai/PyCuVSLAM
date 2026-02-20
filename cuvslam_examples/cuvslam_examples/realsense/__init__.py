from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from cuvslam_examples.realsense.utils import Pose


@dataclass
class TrackingResult:
    """Single tracking frame output from a pipeline."""

    timestamp: int
    vo_pose: Pose
    slam_pose: Pose
    images: Tuple[np.ndarray, ...]
    landmarks: List = field(default_factory=list)
