from typing import Union, Optional
import torch
import numpy as np


def noise(
    x: torch.Tensor,
    y: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    *,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor: ...

def noise2(
    x: Union[torch.Tensor, np.ndarray, float],
    y: Union[torch.Tensor, np.ndarray, float],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor: ...

def noise3(
    x: Union[torch.Tensor, np.ndarray, float],
    y: Union[torch.Tensor, np.ndarray, float],
    z: Union[torch.Tensor, np.ndarray, float],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor: ...
