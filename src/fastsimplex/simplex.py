
import os
import platform
from typing import TYPE_CHECKING, Protocol, Optional

import torch
import numpy as np
from torch.utils.cpp_extension import load

# Type definitions for Pyright
if TYPE_CHECKING:
    class SimplexModuleProtocol(Protocol):
        def simplex(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            z: torch.Tensor,
            octaves: int,
            persistence: float,
            lacunarity: float,
            seed: int
        ) -> torch.Tensor: ...

    _simplex: SimplexModuleProtocol
else:
    # Detect the platform and available backends
    is_macos = platform.system() == "Darwin"
    has_cuda = torch.cuda.is_available()

    # Choose which extensions to build based on platform
    extension_sources = []
    extra_cflags = []
    extra_ldflags = []
    extra_cuda_cflags = []

    # Build path
    sources_dir = os.path.dirname(__file__)

    if is_macos:
        # On macOS, try to use Metal
        extra_cflags.extend(["-std=c++17", "-DWITH_METAL", "-mmacosx-version-min=11.0", "-fobjc-arc"])
        extension_sources.append(os.path.join(sources_dir, "_simplex_metal.mm"))
        extra_ldflags.extend([
            "-framework", "Metal",
            "-framework", "Foundation",
            "-mmacosx-version-min=11.0",
        ])

    if has_cuda:
        # If CUDA is available, also include CUDA sources
        extension_sources.append(os.path.join(sources_dir, "_simplex.cu"))
        extra_cuda_cflags.append("-DWITH_CUDA")
        extra_cuda_cflags.append("-arch=sm_70")  # Set appropriate CUDA architecture

    # Include the dispatcher if we have multiple backends
    if len(extension_sources) > 1:
        extension_sources.append(os.path.join(sources_dir, "_simplex_dispatch.cpp"))
        extra_cflags.append("-std=c++14")  # For the dispatcher

    # If no backend is available, default to CPU implementation or raise error
    if not extension_sources:
        raise RuntimeError("No supported backend available. Requires either macOS for Metal or CUDA support.")

    # Load the extension based on available backends
    _simplex = load(
        name="simplex",
        sources=extension_sources,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )


def noise2(
    x: torch.Tensor | np.ndarray | float,
    y: torch.Tensor | np.ndarray | float,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Generate 2D simplex noise.

    Args:
        x (torch.Tensor | np.ndarray | float): x coordinates
        y (torch.Tensor | np.ndarray | float): y coordinates
        octaves (int, optional): Number of octaves. Defaults to 1.
        persistence (float, optional): Amplitude decrease factor per octave. Defaults to 0.5.
        lacunarity (float, optional): Frequency increase factor per octave. Defaults to 2.0.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        torch.Tensor: Generated simplex noise tensor matching input shape.
    """
    with torch.no_grad():
        # Convert to torch tensors if necessary.
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Ensure inputs are at least 1D tensors
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)

        # Flatten
        X_flat = x.reshape(-1).contiguous()
        Y_flat = y.reshape(-1).contiguous()

        # Run kernel
        noise_flat = _simplex.simplex(
            X_flat,
            Y_flat,
            torch.tensor([], dtype=torch.float32),
            octaves,
            persistence,
            lacunarity,
            seed,
        )
        return noise_flat.reshape(x.shape)


def noise3(
    x: torch.Tensor | np.ndarray | float,
    y: torch.Tensor | np.ndarray | float,
    z: torch.Tensor | np.ndarray | float,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Generate 3D simplex noise.

    Args:
        x (torch.Tensor | np.ndarray | float): x coordinates
        y (torch.Tensor | np.ndarray | float): y coordinates
        z (torch.Tensor | np.ndarray | float): z coordinates
        octaves (int, optional): Number of octaves. Defaults to 1.
        persistence (float, optional): Amplitude decrease factor per octave. Defaults to 0.5.
        lacunarity (float, optional): Frequency increase factor per octave. Defaults to 2.0.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        torch.Tensor: Generated 3D simplex noise.
    """
    with torch.no_grad():
        # Convert to torch tensors if necessary.
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)

        # Ensure inputs are at least 1D tensors
        if x.dim() == 0:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        if z.dim() == 0:
            z = z.unsqueeze(0)

        # Flatten
        X_flat = x.reshape(-1)
        Y_flat = y.reshape(-1)
        Z_flat = z.reshape(-1)

        # Run kernel
        noise_flat = _simplex.simplex(
            X_flat, Y_flat, Z_flat, octaves, persistence, lacunarity, seed
        )
        return noise_flat.reshape(x.shape)


def noise(
    x: torch.Tensor,
    y: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    *,
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int = 0,
) -> torch.Tensor:
    """Generate 2D or 3D simplex noise.

    Args:
        x (torch.Tensor): x coordinates
        y (torch.Tensor): y coordinates
        z (torch.Tensor, optional): z coordinates. Defaults to None.
        octaves (int, optional): Number of octaves. Defaults to 1.
        persistence (float, optional): Amplitude decrease factor per octave. Defaults to 0.5.
        lacunarity (float, optional): Frequency increase factor per octave. Defaults to 2.0.
        seed (int, optional): Random seed. Defaults to 0.

    Returns:
        torch.Tensor: Generated simplex noise tensor matching input shape.

    Example:
        >>> x = torch.linspace(-1, 1, 10)
        >>> y = torch.linspace(-1, 1, 10)
        >>> X, Y = torch.meshgrid(x, y, indexing="ij")
        >>> noise_2d = noise(X, Y, octaves=4)  # 2D noise
        >>> Z = torch.linspace(-1, 1, 10)
        >>> X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        >>> noise_3d = noise(X, Y, Z, octaves=4)  # 3D noise
    """
    with torch.no_grad():
        if z is None:
            return noise2(
                x,
                y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                seed=seed,
            )
        else:
            return noise3(
                x,
                y,
                z,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                seed=seed,
            )
