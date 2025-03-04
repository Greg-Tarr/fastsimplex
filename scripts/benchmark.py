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


def detect_backend():
    """Detect which GPU backend (if any) is being used."""
    if torch.cuda.is_available():
        return "CUDA"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "MPS"
    else:
        return "CPU"


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


def generate_noise_array_gpu_2d(origin, shape, noise_cfg: NoiseConfig) -> np.ndarray:
    """
    Generate a 2D noise array using the GPU-accelerated noise generator.
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


def generate_noise_array_gpu_3d(origin, shape, noise_cfg: NoiseConfig) -> np.ndarray:
    """
    Generate a 3D noise array using the GPU-accelerated noise generator.
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
    return noise_tensor.reshape(shape).cpu().numpy()


def benchmark_2d(noise_cfg: NoiseConfig, shape=(256, 256), iterations=3, backend=""):
    print(f"\n[2D] Config: {noise_cfg} | Shape: {shape}")
    default_noise, gpu_noise = None, None

    # Warm-up calls
    _ = generate_noise_array_default_2d((0, 0), shape, noise_cfg)
    _ = generate_noise_array_gpu_2d((0, 0), shape, noise_cfg)

    # Benchmark default noise
    start_default = time.time()
    for _ in range(iterations):
        default_noise = generate_noise_array_default_2d((0, 0), shape, noise_cfg)
    t_default = (time.time() - start_default) / iterations

    # Benchmark GPU noise
    start_gpu = time.time()
    for _ in range(iterations):
        gpu_noise = generate_noise_array_gpu_2d((0, 0), shape, noise_cfg)
    t_gpu = (time.time() - start_gpu) / iterations

    # Calculate speedup
    speedup = t_default / t_gpu

    print(f"Default 2D avg time: {t_default:.4f}s, {backend} 2D avg time: {t_gpu:.4f}s")
    print(f"Speedup with {backend}: {speedup:.2f}x")

    return default_noise, gpu_noise, t_default, t_gpu, speedup


def benchmark_3d(noise_cfg: NoiseConfig, shape=(64, 64, 64), iterations=3, backend=""):
    print(f"\n[3D] Config: {noise_cfg} | Shape: {shape}")
    default_noise, gpu_noise = None, None

    # Warm-up calls
    _ = generate_noise_array_default_3d((0, 0, 0), shape, noise_cfg)
    _ = generate_noise_array_gpu_3d((0, 0, 0), shape, noise_cfg)

    # Benchmark default noise
    start_default = time.time()
    for _ in range(iterations):
        default_noise = generate_noise_array_default_3d((0, 0, 0), shape, noise_cfg)
    t_default = (time.time() - start_default) / iterations

    # Benchmark GPU noise
    start_gpu = time.time()
    for _ in range(iterations):
        gpu_noise = generate_noise_array_gpu_3d((0, 0, 0), shape, noise_cfg)
    t_gpu = (time.time() - start_gpu) / iterations

    # Calculate speedup
    speedup = t_default / t_gpu

    print(f"Default 3D avg time: {t_default:.4f}s, {backend} 3D avg time: {t_gpu:.4f}s")
    print(f"Speedup with {backend}: {speedup:.2f}x")

    return default_noise, gpu_noise, t_default, t_gpu, speedup


def plot_comparison_2d(default_noise, gpu_noise, title, filename, backend):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(default_noise, cmap="gray", origin="lower")
    axs[0].set_title("Default 2D Noise")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(gpu_noise, cmap="gray", origin="lower")
    axs[1].set_title(f"{backend} 2D Noise")
    fig.colorbar(im1, ax=axs[1])
    fig.suptitle(title)
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def plot_comparison_3d(default_noise, gpu_noise, title, filename, backend):
    # For 3D noise, take the middle slice along the first axis.
    mid = default_noise.shape[0] // 2
    default_slice = default_noise[mid, :, :]
    gpu_slice = gpu_noise[mid, :, :]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im0 = axs[0].imshow(default_slice, cmap="gray", origin="lower")
    axs[0].set_title("Default 3D Noise (Slice)")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(gpu_slice, cmap="gray", origin="lower")
    axs[1].set_title(f"{backend} 3D Noise (Slice)")
    fig.colorbar(im1, ax=axs[1])
    fig.suptitle(title)
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def save_summary(results, output_dir, backend):
    # Save summary text file with benchmark results
    summary_path = os.path.join(output_dir, f"benchmark_results_{backend.lower()}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"FastSimplex Benchmark Results using {backend}\n")
        f.write("=" * 50 + "\n\n")

        # 2D Results
        f.write("2D NOISE BENCHMARKS\n")
        f.write("-" * 30 + "\n")
        for result in results['2d']:
            cfg, shape, t_default, t_gpu, speedup = result
            f.write(f"Config: octaves={cfg.octaves}, scale={cfg.scale}, "
                    f"persistence={cfg.persistence}, lacunarity={cfg.lacunarity}\n")
            f.write(f"Shape: {shape}\n")
            f.write(f"Default time: {t_default:.4f}s, {backend} time: {t_gpu:.4f}s\n")
            f.write(f"Speedup: {speedup:.2f}x\n\n")

        # 3D Results
        f.write("\n3D NOISE BENCHMARKS\n")
        f.write("-" * 30 + "\n")
        for result in results['3d']:
            cfg, shape, t_default, t_gpu, speedup = result
            f.write(f"Config: octaves={cfg.octaves}, scale={cfg.scale}, "
                    f"persistence={cfg.persistence}, lacunarity={cfg.lacunarity}\n")
            f.write(f"Shape: {shape}\n")
            f.write(f"Default time: {t_default:.4f}s, {backend} time: {t_gpu:.4f}s\n")
            f.write(f"Speedup: {speedup:.2f}x\n\n")

    print(f"Saved benchmark summary to {summary_path}")


def main():
    # Detect which backend is being used
    backend = detect_backend()
    print(f"\nRunning benchmarks using {backend} acceleration\n")

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../assets")
    os.makedirs(output_dir, exist_ok=True)

    # Define parameter combinations
    scales = [25.0, 50.0]
    octaves_list = [1, 3, 5, 9]
    persistence = 0.5
    lacunarity = 2.0

    # Store results for summary
    benchmark_results = {
        '2d': [],
        '3d': []
    }

    # --- 2D noise tests ---
    for scale in scales:
        for octaves in octaves_list:
            cfg = NoiseConfig(
                octaves=octaves,
                scale=scale,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            default2d, gpu2d, t_def, t_gpu, speedup = benchmark_2d(
                cfg, shape=(256, 256), iterations=10, backend=backend
            )

            # Store results
            benchmark_results['2d'].append((cfg, (256, 256), t_def, t_gpu, speedup))

            title = (
                f"2D Noise - Scale: {scale}, Octaves: {octaves}\n"
                f"Default avg: {t_def:.4f}s, {backend} avg: {t_gpu:.4f}s (Speedup: {speedup:.2f}x)"
            )
            filename = os.path.join(output_dir, f"2d_scale{scale}_octaves{octaves}_{backend.lower()}.jpg")
            plot_comparison_2d(default2d, gpu2d, title, filename, backend)

    # --- 3D noise tests ---
    for scale in scales:
        for octaves in octaves_list:
            cfg = NoiseConfig(
                octaves=octaves,
                scale=scale,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            default3d, gpu3d, t_def, t_gpu, speedup = benchmark_3d(
                cfg, shape=(128, 128, 128), iterations=5, backend=backend
            )

            # Store results
            benchmark_results['3d'].append((cfg, (128, 128, 128), t_def, t_gpu, speedup))

            title = (
                f"3D Noise (Slice) - Scale: {scale}, Octaves: {octaves}\n"
                f"Default avg: {t_def:.4f}s, {backend} avg: {t_gpu:.4f}s (Speedup: {speedup:.2f}x)"
            )
            filename = os.path.join(output_dir, f"3d_scale{scale}_octaves{octaves}_{backend.lower()}.jpg")
            plot_comparison_3d(default3d, gpu3d, title, filename, backend)

    # Save summary of benchmark results
    save_summary(benchmark_results, output_dir, backend)


if __name__ == "__main__":
    main()
