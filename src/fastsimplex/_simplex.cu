#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

struct NoiseParams {
    int octaves;
    float persistence;
    float lacunarity;
    int seed;
};

// Hash function
__device__ int hash(int x, int y, int z, int seed) {
    int h = seed + x * 374761393 + y * 668265263 + z * 5432646921;
    h = (h ^ (h >> 13)) * 1274126177;
    return h ^ (h >> 16);
}

__device__ float gradient2d(int hash, float x, float y) {
    // Convert low 3 bits of hash to 8 gradient directions
    int h = hash & 7;
    float u = (h < 4) ? x : y;
    float v = (h < 4) ? y : x;
    return ((h & 1) ? -u : u) + ((h & 2) ? -2.0f * v : 2.0f * v);
}

__device__ float gradient3d(int hash, float x, float y, float z) {
    // Convert low 4 bits of hash to 12 gradient directions
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

// 2D Simplex noise
__device__ float snoise2d(float x, float y, int seed) {
    // Apply seed offset
    x += seed % 1000;
    y += (seed / 1000) % 1000;

    const float F2 = 0.366025404f; // 0.5 * (sqrt(3) - 1)
    const float G2 = 0.211324865f; // (3 - sqrt(3)) / 6

    // Skew the input space to determine cell
    float s = (x + y) * F2;
    float xs = x + s;
    float ys = y + s;
    int i = floorf(xs);
    int j = floorf(ys);

    // Unskew the cell origin back to (x,y) space
    float t = (float)(i + j) * G2;
    float X0 = i - t;
    float Y0 = j - t;
    float x0 = x - X0;
    float y0 = y - Y0;

    // Determine which simplex we're in
    int i1, j1;
    if (x0 > y0) { // lower triangle, XY order: (0,0)->(1,0)->(1,1)
        i1 = 1; j1 = 0;
    } else {       // upper triangle, YX order: (0,0)->(0,1)->(1,1)
        i1 = 0; j1 = 1;
    }

    // Calculate other corners
    float x1 = x0 - i1 + G2;
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;

    // Calculate contribution from three corners
    float n0, n1, n2;

    // Calculate kernel function
    t = 0.5f - x0 * x0 - y0 * y0;
    if (t < 0.0f) {
        n0 = 0.0f;
    } else {
        t *= t;
        n0 = t * t * gradient2d(hash(i, j, 0, seed), x0, y0);
    }

    t = 0.5f - x1 * x1 - y1 * y1;
    if (t < 0.0f) {
        n1 = 0.0f;
    } else {
        t *= t;
        n1 = t * t * gradient2d(hash(i + i1, j + j1, 0, seed), x1, y1);
    }

    t = 0.5f - x2 * x2 - y2 * y2;
    if (t < 0.0f) {
        n2 = 0.0f;
    } else {
        t *= t;
        n2 = t * t * gradient2d(hash(i + 1, j + 1, 0, seed), x2, y2);
    }

    // Add contributions to get final noise value between -1 and 1
    return 70.0f * (n0 + n1 + n2);
}

// 3D Simplex noise
__device__ float snoise3d(float x, float y, float z, int seed) {
    // Apply seed offset
    x += seed % 1000;
    y += (seed / 1000) % 1000;
    z += (seed / 1000000) % 1000;

    const float F3 = 0.333333333f; // 1/3
    const float G3 = 0.166666667f; // 1/6

    // Skew the input space to determine cell
    float s = (x + y + z) * F3;
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    int i = floorf(xs);
    int j = floorf(ys);
    int k = floorf(zs);

    // Unskew the cell origin back to (x,y,z) space
    float t = (float)(i + j + k) * G3;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    // Determine which simplex we're in
    int i1, j1, k1; // Offsets for second corner of simplex
    int i2, j2, k2; // Offsets for third corner of simplex

    if (x0 >= y0) {
        if (y0 >= z0) { // X Y Z order
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        } else if (x0 >= z0) { // X Z Y order
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
        } else { // Z X Y order
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
        }
    } else { // x0 < y0
        if (y0 < z0) { // Z Y X order
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
        } else if (x0 < z0) { // Y Z X order
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
        } else { // Y X Z order
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        }
    }

    // Calculate corners coordinates
    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3;
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    // Calculate noise contributions from each corner
    float n0, n1, n2, n3;

    // Calculate kernel function
    t = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t < 0.0f) {
        n0 = 0.0f;
    } else {
        t *= t;
        n0 = t * t * gradient3d(hash(i, j, k, seed), x0, y0, z0);
    }

    t = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t < 0.0f) {
        n1 = 0.0f;
    } else {
        t *= t;
        n1 = t * t * gradient3d(hash(i + i1, j + j1, k + k1, seed), x1, y1, z1);
    }

    t = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t < 0.0f) {
        n2 = 0.0f;
    } else {
        t *= t;
        n2 = t * t * gradient3d(hash(i + i2, j + j2, k + k2, seed), x2, y2, z2);
    }

    t = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t < 0.0f) {
        n3 = 0.0f;
    } else {
        t *= t;
        n3 = t * t * gradient3d(hash(i + 1, j + 1, k + 1, seed), x3, y3, z3);
    }

    // Add contributions from each corner to get the final noise value between -1 and 1
    return 32.0f * (n0 + n1 + n2 + n3);
}

// Fractal Brownian Motion (FBM) for 2D noise
__global__ void kernel_noise2d(
    const float* x,
    const float* y,
    float* out,
    const NoiseParams params,
    size_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xf = x[idx];
        float yf = y[idx];

        float noise = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float maxAmplitude = 0.0f;

        for (int i = 0; i < params.octaves; i++) {
            float n = snoise2d(xf * frequency, yf * frequency, params.seed);
            noise += n * amplitude;
            maxAmplitude += amplitude;
            amplitude *= params.persistence;
            frequency *= params.lacunarity;
        }

        noise = noise / maxAmplitude;
        out[idx] = noise;
    }
}

// Fractal Brownian Motion (FBM) for 3D noise
__global__ void kernel_noise3d(
    const float* x,
    const float* y,
    const float* z,
    float* out,
    const NoiseParams params,
    size_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xf = x[idx];
        float yf = y[idx];
        float zf = z[idx];

        float noise = 0.0f;
        float amplitude = 1.0f;
        float frequency = 1.0f;
        float maxAmplitude = 0.0f;

        for (int i = 0; i < params.octaves; i++) {
            float n = snoise3d(xf * frequency, yf * frequency, zf * frequency, params.seed);
            noise += n * amplitude;
            maxAmplitude += amplitude;
            amplitude *= params.persistence;
            frequency *= params.lacunarity;
        }

        noise = noise / maxAmplitude;
        out[idx] = noise;
    }
}

torch::Tensor simplex_cuda(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor z,
    int octaves,
    float persistence,
    float lacunarity,
    int seed
) {
    // Check if we are running 2D or 3D
    bool is3D = (z.numel() > 0);

    // Ensure tensors are on CUDA
    x = x.to(torch::kCUDA).contiguous();
    y = y.to(torch::kCUDA).contiguous();
    if (is3D) {
        z = z.to(torch::kCUDA).contiguous();
    }

    // Get the total number of elements
    auto count = x.numel();

    // Create an output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out = torch::empty({count}, options);

    // Setup NoiseParams struct
    NoiseParams params;
    params.octaves = octaves;
    params.persistence = persistence;
    params.lacunarity = lacunarity;
    params.seed = seed;

    // Calculate launch configuration
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    // Launch the appropriate kernel
    if (is3D) {
        kernel_noise3d<<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            z.data_ptr<float>(),
            out.data_ptr<float>(),
            params,
            count
        );
    } else {
        kernel_noise2d<<<blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            out.data_ptr<float>(),
            params,
            count
        );
    }

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    return out.reshape(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simplex", &simplex_cuda, "Simplex noise (CUDA)");
}
