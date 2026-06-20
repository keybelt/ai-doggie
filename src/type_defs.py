"""Contains the custom type definitions of the pipeline.

Example:
    >>> from type_defs import Frame
"""

import queue

import numpy as np
from jaxtyping import UInt8

type Frame = UInt8[np.ndarray, "H W C"]
type FrameQueue = queue.Queue[Frame]
type FramePackage = tuple[Frame, bool]

type ParsedMacro = list[tuple[int, int]]
