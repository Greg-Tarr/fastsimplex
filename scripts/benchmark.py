"""
Benchmark Results:

[2D] Config: NoiseConfig(octaves=1, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0275s, Metal 2D avg time: 0.0068s

[2D] Config: NoiseConfig(octaves=3, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0296s, Metal 2D avg time: 0.0073s

[2D] Config: NoiseConfig(octaves=5, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0312s, Metal 2D avg time: 0.0070s

[2D] Config: NoiseConfig(octaves=12, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0358s, Metal 2D avg time: 0.0068s

[2D] Config: NoiseConfig(octaves=1, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0283s, Metal 2D avg time: 0.0069s

[2D] Config: NoiseConfig(octaves=3, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0301s, Metal 2D avg time: 0.0078s

[2D] Config: NoiseConfig(octaves=5, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0314s, Metal 2D avg time: 0.0077s

[2D] Config: NoiseConfig(octaves=12, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (256, 256)
Default 2D avg time: 0.0362s, Metal 2D avg time: 0.0077s

[3D] Config: NoiseConfig(octaves=1, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.0760s, Metal 3D avg time: 0.0126s

[3D] Config: NoiseConfig(octaves=3, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.1834s, Metal 3D avg time: 0.0125s

[3D] Config: NoiseConfig(octaves=5, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.2826s, Metal 3D avg time: 0.0139s

[3D] Config: NoiseConfig(octaves=12, scale=25.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.6118s, Metal 3D avg time: 0.0177s

[3D] Config: NoiseConfig(octaves=1, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.0705s, Metal 3D avg time: 0.0137s

[3D] Config: NoiseConfig(octaves=3, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.1738s, Metal 3D avg time: 0.0130s

[3D] Config: NoiseConfig(octaves=5, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.2826s, Metal 3D avg time: 0.0143s

[3D] Config: NoiseConfig(octaves=12, scale=50.0, persistence=0.5, lacunarity=2.0, seed=0) | Shape: (128, 128, 128)
Default 3D avg time: 1.6336s, Metal 3D avg time: 0.0165s
"""

import os
import time
import numpy as np
import torch
import noise
import matplotlib.pyplot as plt
from dataclasses import dataclass

from fastsimplex import noise2, noise3


@dataclass
class NoiseConfig:
    octaves: int
    scale: float
    persistence: float
    lacunarity: float
    seed: int = 0


def generate_noise_array_default_2d(
    origin, shape, noise_cfg: NoiseConfig
) -> np.ndarray:
    """
    Generate a 2D noise array using the default noise library's snoise2 function.
    """
    x_offset = noise_cfg.seed
    y_offset = noise_cfg.seed
    # Create 1D coordinate arrays with seeding offsets
    x = np.linspace(origin[0] + x_offset, origin[0] + x_offset + shape[1] - 1, shape[1])
    y = np.linspace(origin[1] + y_offset, origin[1] + y_offset + shape[0] - 1, shape[0])
    noise_vals = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Scale coordinates
            x_val = x[j] / noise_cfg.scale
            y_val = y[i] / noise_cfg.scale
            noise_vals[i, j] = noise.snoise2(
                x_val,
                y_val,
                octaves=noise_cfg.octaves,
                persistence=noise_cfg.persistence,
                lacunarity=noise_cfg.lacunarity,
            )
    return noise_vals


def generate_noise_array_default_3d(
    origin, shape, noise_cfg: NoiseConfig
) -> np.ndarray:
    """
    Generate a 3D noise array using the default noise library's snoise3 function.
    (This implementation uses triple nested loops and may be slow for larger grids.)
    """
    x_offset = noise_cfg.seed
    y_offset = noise_cfg.seed
    z_offset = noise_cfg.seed
    x = np.linspace(origin[2] + x_offset, origin[2] + x_offset + shape[2] - 1, shape[2])
    y = np.linspace(origin[1] + y_offset, origin[1] + y_offset + shape[1] - 1, shape[1])
    z = np.linspace(origin[0] + z_offset, origin[0] + z_offset + shape[0] - 1, shape[0])
    noise_vals = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x_val = x[k] / noise_cfg.scale
                y_val = y[j] / noise_cfg.scale
                z_val = z[i] / noise_cfg.scale
                noise_vals[i, j, k] = noise.snoise3(
                    x_val,
                    y_val,
                    z_val,
                    octaves=noise_cfg.octaves,
                    persistence=noise_cfg.persistence,
                    lacunarity=noise_cfg.lacunarity,
                )
    return noise_vals


def generate_noise_array_metal_2d(origin, shape, noise_cfg: NoiseConfig) -> np.ndarray:
    """
    Generate a 2D noise array using the Metal-accelerated noise generator.
    """
    x_offset = noise_cfg.seed
    y_offset = noise_cfg.seed
    x = torch.linspace(
        origin[0] + x_offset,
        origin[0] + x_offset + shape[1] - 1,
        shape[1],
        dtype=torch.float32,
    )
    y = torch.linspace(
        origin[1] + y_offset,
        origin[1] + y_offset + shape[0] - 1,
        shape[0],
        dtype=torch.float32,
    )
    X, Y = torch.meshgrid(x, y, indexing="ij")
    X = X / noise_cfg.scale
    Y = Y / noise_cfg.scale
    noise_tensor = noise2(
        X,
        Y,
        octaves=noise_cfg.octaves,
        persistence=noise_cfg.persistence,
        lacunarity=noise_cfg.lacunarity,
    )
    return noise_tensor.cpu().numpy()


def generate_noise_array_metal_3d(origin, shape, noise_cfg: NoiseConfig) -> np.ndarray:
    """
    Generate a 3D noise array using the Metal-accelerated noise generator.
    """
    x_offset = noise_cfg.seed
    y_offset = noise_cfg.seed
    z_offset = noise_cfg.seed

    # Create linear spaces for each dimension
    x = (
        torch.linspace(
            origin[0] + x_offset,
            origin[0] + x_offset + shape[0] - 1,
            shape[0],
            dtype=torch.float32,
        )
        / noise_cfg.scale
    )

    y = (
        torch.linspace(
            origin[1] + y_offset,
            origin[1] + y_offset + shape[1] - 1,
            shape[1],
            dtype=torch.float32,
        )
        / noise_cfg.scale
    )

    z = (
        torch.linspace(
            origin[2] + z_offset,
            origin[2] + z_offset + shape[2] - 1,
            shape[2],
            dtype=torch.float32,
        )
        / noise_cfg.scale
    )

    # Create meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # Flatten each coordinate array
    X_flat = X.reshape(-1)
    Y_flat = Y.reshape(-1)
    Z_flat = Z.reshape(-1)

    # Call noise function with separate coordinates
    noise_tensor = noise3(
        X_flat,
        Y_flat,
        Z_flat,
        octaves=noise_cfg.octaves,
        persistence=noise_cfg.persistence,
        lacunarity=noise_cfg.lacunarity,
    )

    # Reshape result back to 3D
    return noise_tensor.reshape(shape)


def benchmark_2d(noise_cfg: NoiseConfig, shape=(256, 256), iterations=3):
    print(f"\n[2D] Config: {noise_cfg} | Shape: {shape}")
    # Warm-up calls
    _ = generate_noise_array_default_2d((0, 0), shape, noise_cfg)
    _ = generate_noise_array_metal_2d((0, 0), shape, noise_cfg)

    # Benchmark default noise
    start_default = time.time()
    for _ in range(iterations):
        default_noise = generate_noise_array_default_2d((0, 0), shape, noise_cfg)
    t_default = (time.time() - start_default) / iterations

    # Benchmark Metal noise
    start_metal = time.time()
    for _ in range(iterations):
        metal_noise = generate_noise_array_metal_2d((0, 0), shape, noise_cfg)
    t_metal = (time.time() - start_metal) / iterations

    print(f"Default 2D avg time: {t_default:.4f}s, Metal 2D avg time: {t_metal:.4f}s")
    return default_noise, metal_noise, t_default, t_metal


def benchmark_3d(noise_cfg: NoiseConfig, shape=(64, 64, 64), iterations=3):
    print(f"\n[3D] Config: {noise_cfg} | Shape: {shape}")
    # Warm-up calls
    _ = generate_noise_array_default_3d((0, 0, 0), shape, noise_cfg)
    _ = generate_noise_array_metal_3d((0, 0, 0), shape, noise_cfg)

    # Benchmark default noise
    start_default = time.time()
    for _ in range(iterations):
        default_noise = generate_noise_array_default_3d((0, 0, 0), shape, noise_cfg)
    t_default = (time.time() - start_default) / iterations

    # Benchmark Metal noise
    start_metal = time.time()
    for _ in range(iterations):
        metal_noise = generate_noise_array_metal_3d((0, 0, 0), shape, noise_cfg)
    t_metal = (time.time() - start_metal) / iterations

    print(f"Default 3D avg time: {t_default:.4f}s, Metal 3D avg time: {t_metal:.4f}s")
    return default_noise, metal_noise, t_default, t_metal


def plot_comparison_2d(default_noise, metal_noise, title, filename):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(default_noise, cmap="gray", origin="lower")
    axs[0].set_title("Default 2D Noise")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(metal_noise, cmap="gray", origin="lower")
    axs[1].set_title("Metal 2D Noise")
    fig.colorbar(im1, ax=axs[1])
    fig.suptitle(title)
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def plot_comparison_3d(default_noise, metal_noise, title, filename):
    # For 3D noise, take the middle slice along the first axis.
    mid = default_noise.shape[0] // 2
    default_slice = default_noise[mid, :, :]
    metal_slice = metal_noise[mid, :, :]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(default_slice, cmap="gray", origin="lower")
    axs[0].set_title("Default 3D Noise (Slice)")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(metal_slice, cmap="gray", origin="lower")
    axs[1].set_title("Metal 3D Noise (Slice)")
    fig.colorbar(im1, ax=axs[1])
    fig.suptitle(title)
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "../assets")
    os.makedirs(output_dir, exist_ok=True)

    # Define parameter combinations
    scales = [25.0, 50.0]
    octaves_list = [1, 3, 5, 9]
    persistence = 0.5
    lacunarity = 2.0

    # --- 2D noise tests ---
    for scale in scales:
        for octaves in octaves_list:
            cfg = NoiseConfig(
                octaves=octaves,
                scale=scale,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            default2d, metal2d, t_def, t_met = benchmark_2d(
                cfg, shape=(256, 256), iterations=10
            )
            title = (
                f"2D Noise - Scale: {scale}, Octaves: {octaves}\n"
                f"Default avg: {t_def:.4f}s, Metal avg: {t_met:.4f}s"
            )
            filename = os.path.join(output_dir, f"2d_scale{scale}_octaves{octaves}.jpg")
            plot_comparison_2d(default2d, metal2d, title, filename)

    # --- 3D noise tests ---
    for scale in scales:
        for octaves in octaves_list:
            cfg = NoiseConfig(
                octaves=octaves,
                scale=scale,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            default3d, metal3d, t_def, t_met = benchmark_3d(
                cfg, shape=(128, 128, 128), iterations=5
            )
            title = (
                f"3D Noise (Slice) - Scale: {scale}, Octaves: {octaves}\n"
                f"Default avg: {t_def:.4f}s, Metal avg: {t_met:.4f}s"
            )
            filename = os.path.join(output_dir, f"3d_scale{scale}_octaves{octaves}.jpg")
            plot_comparison_3d(default3d, metal3d, title, filename)


if __name__ == "__main__":
    main()
