import torch
import numpy as np
from fastsimplex import noise, noise2, noise3


def test_noise2d_basic():
    """Test basic 2D noise generation with single values"""
    result = noise2(0.0, 0.0)
    assert isinstance(
        result, torch.Tensor
    ), f"Expected torch.Tensor, got {type(result)}"
    assert result.dim() == 1, f"Expected dim 1, got {result.dim()}"
    assert result.size(0) == 1, f"Expected size 1, got {result.size(0)}"

    # Test with numpy values
    result_np = noise2(np.float32(0.0), np.float32(0.0))
    assert isinstance(
        result_np, torch.Tensor
    ), f"Expected torch.Tensor, got {type(result_np)}"

    # Test with different seeds produce different results
    n1 = noise2(1.0, 0.0, seed=1)
    n2 = noise2(1.0, 0.0, seed=2)
    assert not torch.allclose(
        n1, n2
    ), f"Seeds 1 and 2 produced same result: {n1} vs {n2}"


def test_noise2d_arrays():
    """Test 2D noise generation with arrays"""
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    result = noise2(X, Y)
    assert result.shape == X.shape, f"Expected shape {X.shape}, got {result.shape}"

    # Test with numpy arrays
    x_np = np.linspace(-1, 1, 10)
    y_np = np.linspace(-1, 1, 10)
    X_np, Y_np = np.meshgrid(x_np, y_np, indexing="ij")
    result_np = noise2(X_np, Y_np)
    assert isinstance(
        result_np, torch.Tensor
    ), f"Expected torch.Tensor, got {type(result_np)}"
    assert (
        result_np.shape == X_np.shape
    ), f"Expected shape {X_np.shape}, got {result_np.shape}"


def test_noise3d_basic():
    """Test basic 3D noise generation"""
    result = noise3(0.0, 0.0, 0.0)
    assert isinstance(
        result, torch.Tensor
    ), f"Expected torch.Tensor, got {type(result)}"
    assert result.dim() == 1, f"Expected dim 1, got {result.dim()}"
    assert result.size(0) == 1, f"Expected size 1, got {result.size(0)}"

    # Test different seeds
    n1 = noise3(1.0, 0.0, 0.0, seed=1)
    n2 = noise3(1.0, 0.0, 0.0, seed=2)
    assert not torch.allclose(
        n1, n2
    ), f"Seeds 1 and 2 produced same result: {n1} vs {n2}"


def test_noise3d_arrays():
    """Test 3D noise generation with arrays"""
    size = 4
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    z = torch.linspace(-1, 1, size)
    # Note: For 3D, we use consistent ordering for meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    result = noise3(X, Y, Z)
    expected_shape = (size, size, size)
    assert (
        result.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {result.shape}"


def test_noise_octaves():
    """Test noise with different octave settings"""
    x = torch.linspace(-1, 1, 4)
    y = torch.linspace(-1, 1, 4)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    n1 = noise(X, Y, octaves=1)
    n2 = noise(X, Y, octaves=4)
    assert not torch.allclose(
        n1, n2
    ), f"Different octaves produced same result: {n1} vs {n2}"

    n3 = noise(X, Y, octaves=4, persistence=0.3, lacunarity=3.0)
    n4 = noise(X, Y, octaves=4, persistence=0.7, lacunarity=1.5)
    assert not torch.allclose(
        n3, n4
    ), f"Different persistence/lacunarity produced same result: {n3} vs {n4}"


def test_value_ranges():
    """Test that noise values stay within expected ranges"""
    size = 100
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    result = noise(X, Y, octaves=4)
    assert torch.all(result >= -1.0), f"Values below -1.0 found: min={result.min()}"
    assert torch.all(result <= 1.0), f"Values above 1.0 found: max={result.max()}"


if __name__ == "__main__":
    # Running tests directly if desired
    test_noise2d_basic()
    test_noise2d_arrays()
    test_noise3d_basic()
    test_noise3d_arrays()
    test_noise_octaves()
    test_value_ranges()
    print("All tests passed!")
