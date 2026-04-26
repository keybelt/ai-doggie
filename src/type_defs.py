"""Contain the custom type defintions of the pipeline.

Example:
    >>> from type_defs import Frame
"""

import queue

import numpy as np
import torch
from jaxtyping import Float32, UInt8

type Frame = UInt8[np.ndarray, "H W C"]
type FrameQueue = queue.Queue[Frame]
type FramePackage = tuple[Frame, bool]

type ConvWeight = Float32[torch.Tensor, "kernel_size kernel_size C_in C_out"]
type ConvBias = Float32[torch.Tensor, "C_out"]

type ParsedMacro = list[tuple[int, int]]
