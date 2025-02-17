#include <metal_stdlib>
using namespace metal;

struct NoiseParams {
    int octaves;
    float persistence;
    float lacunarity;
    int seed;
};

// --- Utility functions --- //

inline float2 mod289(float2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float3 mod289(float3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float4 mod289(float4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

inline float2 permute(float2 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}

inline float3 permute(float3 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}

inline float4 permute(float4 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}

// --- 2D Simplex noise --- //

inline float snoise2d(float2 v, int seed) {
    // Incorporate a seed offset.
    v += float2(float(seed % 1000), float((seed / 1000) % 1000));

    const float F2 = 0.3660254037844386;  // (sqrt(3)-1)/2
    const float G2 = 0.21132486540518713; // (3-sqrt(3))/6
    float2 i = floor(v + (v.x + v.y) * F2);
    float2 x0 = v - i + (i.x + i.y) * G2;

    float2 i1 = (x0.x > x0.y) ? float2(1.0, 0.0) : float2(0.0, 1.0);

    float2 x1 = x0 - i1 + G2;
    float2 x2 = x0 - 1.0 + 2.0 * G2;

    i = mod289(i);
    float3 p = permute( permute( i.y + float3(0.0, i1.y, 1.0))
                       + i.x + float3(0.0, i1.x, 1.0));

    // Map p to gradients.
    float3 j = p - 49.0 * floor(p / 49.0); // mod 7*7
    float3 x_grad = floor(j / 7.0);
    float3 y_grad = j - 7.0 * x_grad;
    x_grad = (x_grad * 2.0 + 0.5) / 7.0 - 1.0;
    y_grad = (y_grad * 2.0 + 0.5) / 7.0 - 1.0;
    float3 norm = rsqrt(x_grad * x_grad + y_grad * y_grad + 1e-6);
    x_grad *= norm;
    y_grad *= norm;

    float3 t = max(0.5 - float3(dot(x0, x0),
                                dot(x1, x1),
                                dot(x2, x2)), 0.0);
    t = t * t;
    t = t * t;

    float3 gradDot;
    gradDot.x = x_grad.x * x0.x + y_grad.x * x0.y;
    gradDot.y = x_grad.y * x1.x + y_grad.y * x1.y;
    gradDot.z = x_grad.z * x2.x + y_grad.z * x2.y;
    return 70.0 * dot(t, gradDot);
}

// --- 3D Simplex noise --- //

inline float snoise3d(float3 v, int seed) {
    // Incorporate a seed offset.
    v += float3(float(seed % 1000), float((seed / 1000) % 1000), float((seed / 1000000) % 1000));

    const float F3 = 1.0 / 3.0;
    const float G3 = 1.0 / 6.0;
    float3 s = floor(v + dot(v, float3(F3)));
    float3 x0 = v - s + dot(s, float3(G3));

    float3 e = step(float3(0.0), x0 - x0.yzx);
    float3 i1 = e * (1.0 - e.zxy);
    float3 i2 = 1.0 - e.zxy * (1.0 - e);

    float3 x1 = x0 - i1 + G3;
    float3 x2 = x0 - i2 + 2.0 * G3;
    float3 x3 = x0 - 1.0 + 3.0 * G3;

    s = mod289(s);
    float4 p = permute( permute( permute(
                 s.z + float4(0.0, i1.z, i2.z, 1.0))
               + s.y + float4(0.0, i1.y, i2.y, 1.0))
               + s.x + float4(0.0, i1.x, i2.x, 1.0));

    float4 j = p - 49.0 * floor(p / 49.0); // mod 7*7

    float4 x_grad = floor(j / 7.0);
    float4 y_grad = j - 7.0 * x_grad;
    x_grad = (x_grad * 2.0 + 0.5) / 7.0 - 1.0;
    y_grad = (y_grad * 2.0 + 0.5) / 7.0 - 1.0;
    float4 z_grad = sqrt(max(1.0 - x_grad * x_grad - y_grad * y_grad, 0.0));

    float4 t0 = max(0.6 - float4(dot(x0, x0),
                                 dot(x1, x1),
                                 dot(x2, x2),
                                 dot(x3, x3)), 0.0);
    t0 = t0 * t0;
    t0 = t0 * t0;

    float4 gradDot;
    gradDot.x = x_grad.x * x0.x + y_grad.x * x0.y + z_grad.x * x0.z;
    gradDot.y = x_grad.y * x1.x + y_grad.y * x1.y + z_grad.y * x1.z;
    gradDot.z = x_grad.z * x2.x + y_grad.z * x2.y + z_grad.z * x2.z;
    gradDot.w = x_grad.w * x3.x + y_grad.w * x3.y + z_grad.w * x3.z;

    return 32.0 * dot(t0, gradDot);
}

// --- Kernels --- //

kernel void kernel_noise2d(const device float *x   [[ buffer(0) ]],
                           const device float *y   [[ buffer(1) ]],
                           device float *out       [[ buffer(2) ]],
                           constant NoiseParams &params [[ buffer(3) ]],
                           uint id                 [[ thread_position_in_grid ]]) {
    float xf = x[id];
    float yf = y[id];

    float noise = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxAmplitude = 0.0;
    for (int i = 0; i < params.octaves; i++) {
        float n = snoise2d(float2(xf * frequency, yf * frequency), params.seed);
        noise += n * amplitude;
        maxAmplitude += amplitude;
        amplitude *= params.persistence;
        frequency *= params.lacunarity;
    }
    noise = noise / maxAmplitude;
    out[id] = noise;
}

kernel void kernel_noise3d(const device float *x   [[ buffer(0) ]],
                           const device float *y   [[ buffer(1) ]],
                           const device float *z   [[ buffer(2) ]],
                           device float *out       [[ buffer(3) ]],
                           constant NoiseParams &params [[ buffer(4) ]],
                           uint id                 [[ thread_position_in_grid ]]) {
    float xf = x[id];
    float yf = y[id];
    float zf = z[id];

    float noise = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxAmplitude = 0.0;
    for (int i = 0; i < params.octaves; i++) {
        float n = snoise3d(float3(xf * frequency, yf * frequency, zf * frequency), params.seed);
        noise += n * amplitude;
        maxAmplitude += amplitude;
        amplitude *= params.persistence;
        frequency *= params.lacunarity;
    }
    noise = noise / maxAmplitude;
    out[id] = noise;
}
