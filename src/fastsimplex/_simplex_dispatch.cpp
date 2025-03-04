#include <torch/extension.h>
#include <stdexcept>
#include <string>

// Forward declarations
#ifdef WITH_CUDA
torch::Tensor simplex_cuda(torch::Tensor x, torch::Tensor y, torch::Tensor z,
                          int octaves, float persistence, float lacunarity, int seed);
#endif

#ifdef WITH_METAL
torch::Tensor simplex_metal(torch::Tensor x, torch::Tensor y, torch::Tensor z,
                           int octaves, float persistence, float lacunarity, int seed);
#endif

// The dispatcher function selects the appropriate backend
torch::Tensor simplex_dispatch(torch::Tensor x, torch::Tensor y, torch::Tensor z,
                              int octaves, float persistence, float lacunarity, int seed) {
    // First, determine which backends are available
    #if defined(WITH_CUDA) && defined(WITH_METAL)
    // Both CUDA and Metal implementations are available, check if CUDA is accessible
    if (torch::cuda::is_available()) {
        return simplex_cuda(x, y, z, octaves, persistence, lacunarity, seed);
    } else {
        return simplex_metal(x, y, z, octaves, persistence, lacunarity, seed);
    }
    #elif defined(WITH_CUDA)
    // Only CUDA is available
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available, but no other backend was compiled");
    }
    return simplex_cuda(x, y, z, octaves, persistence, lacunarity, seed);
    #elif defined(WITH_METAL)
    // Only Metal is available
    return simplex_metal(x, y, z, octaves, persistence, lacunarity, seed);
    #else
    // No backend available
    throw std::runtime_error("No backend was compiled (neither CUDA nor Metal)"); // TODO: cpu fallback implementation with SIMD
    #endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simplex", &simplex_dispatch, "Accelerated simplex noise.");
}
